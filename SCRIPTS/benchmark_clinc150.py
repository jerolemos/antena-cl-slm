#!/usr/bin/env python3
"""
ANTENA Mega-Test Final — CLINC150 (LNT) + Inteligencia Zero-Shot
=============================================================================
Este script implementa el benchmark definitivo de la investigación.
Fase 1: Baseline de Inteligencia (PIQA) y Lenguaje (Perplejidad WikiText-2).
Fase 2: Lectura de Hiperparámetros (champions_config.json).
Fase 3: Entrenamiento Masivo en CLINC150 (Zero, IL2A, Rep10, Rep50, Rep100).
Fase 4: Evaluación Post-Entrenamiento de Inteligencia y Lenguaje.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
import numpy as np
import pandas as pd
import gc

# Dependencias instaladas externamente en el Cuaderno

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    lm_eval = None
    HFLM = None

# Monkeypatch for torchvision.io.VideoReader bug in Colab datasets torch formatting
try:
    import torchvision.io
    if not hasattr(torchvision.io, "VideoReader") or not isinstance(torchvision.io.VideoReader, type):
        class VideoReader: pass
        torchvision.io.VideoReader = VideoReader
except Exception:
    pass

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CORE"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "CORE"))
try:
    from antena_gestalt_core import EGPRRegulator, LayerGestalt
    from antena_latent_il2a import MidLayerGestalt, compute_il2a_kl_loss
    from antena_paso2_lora_ogp import LoRALinear, LoRAGestalt, inject_lora, remove_lora
except ImportError:
    print("Error: No se encontró CORE/antena_gestalt_core.py, antena_latent_il2a.py o antena_paso2_lora_ogp.py")
    sys.exit(1)

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS_PER_TASK = 5
NUM_CLASSES = 150
NUM_TASKS = 15
HEBBIAN_DECAY = 0.95
TARGET_LAYERS = list(range(10, 30))
RANK = 16
LORA_LR = 5e-4
SCORE_LR = 1e-4
LWF_T = 2.0
MAX_PARALLEL_WORKERS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "RESULTS"
import os
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_FILE = f"{RESULTS_DIR}/SUMMARY.log"

def log_summary(msg):
    """Escribe en el log compacto SUMMARY.log (legible sin desperdiciar tokens)."""
    print(msg)
    with open(SUMMARY_FILE, 'a') as f:
        f.write(msg + "\n")

# =============================================================================
# 2. HERRAMIENTAS ZERO-SHOT Y LENGUAJE (PIQA / WIKITEXT)
# =============================================================================
def evaluate_perplexity(causal_model, tokenizer):
    print("      [Evaluando Perplejidad en WikiText-2]...")
    causal_model.eval()
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            text = response.read().decode('utf-8')
    except Exception as e:
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
            text = "\n\n".join(ds["text"])
        except:
            try:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                text = "\n\n".join(ds["text"])
            except:
                return 999.99
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = 512
    nlls = []
    with torch.no_grad():
        for i in range(0, min(encodings.input_ids.size(1), 512 * 20), seq_len):
            input_ids = encodings.input_ids[:, i:i+seq_len].to(device)
            target_ids = input_ids.clone()
            if input_ids.size(1) == 0: break
            outputs = causal_model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * input_ids.size(1))
    ppl = torch.exp(torch.stack(nlls).sum() / min(encodings.input_ids.size(1), 512 * 20)).item()
    return ppl

def evaluate_piqa(causal_model, tokenizer):
    print("      [Evaluando PIQA Zero-Shot via lm-eval]...")
    lm_obj = HFLM(pretrained=causal_model, tokenizer=tokenizer, batch_size=4)
    results = lm_eval.simple_evaluate(model=lm_obj, tasks=["piqa"], num_fewshot=0)
    return results['results']['piqa']['acc,none'] * 100.0

def evaluate_intelligence(classification_model, tokenizer):
    """
    Truco arquitectónico: Transfiere los pesos del Transformer entrenado 
    a un modelo CausalLM para poder usar su LM_Head pre-entrenada original.
    """
    print("    -> Configurando Modelo CausalLM para Zero-Shot...")
    causal_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    # Transferir pesos de la columna vertebral (ignorando la cabeza de clasificación)
    causal_model.model.load_state_dict(classification_model.model.state_dict())
    
    ppl = evaluate_perplexity(causal_model, tokenizer)
    piqa_acc = evaluate_piqa(causal_model, tokenizer)
    
    del causal_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"Perplexity": ppl, "PIQA": piqa_acc}

# =============================================================================
# 3. MÓDULOS ANTENA Y MEMORIA (IL2A, REPLAY)
# =============================================================================
class GestaltRegulatorNoHead:
    def __init__(self, model, rho_topo=0.8):
        self.model = model
        self.layer_gestalts = []
        self.hooks = []
        self.is_calibrating = False
        self.current_mask = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'score' in name: continue
                lg = LayerGestalt(module, rho=rho_topo)
                self.layer_gestalts.append(lg)
                def hook_fn(lg_ref):
                    def fn(mod, inp, out):
                        if self.is_calibrating: lg_ref.accumulate(inp[0], mask=self.current_mask)
                    return fn
                self.hooks.append(module.register_forward_hook(hook_fn(lg)))
    def start_calibration(self): self.is_calibrating = True
    def end_calibration(self):
        self.is_calibrating = False
        for lg in self.layer_gestalts: lg.fit(hebbian_decay=HEBBIAN_DECAY)
    def project_all_gradients(self):
        for lg in self.layer_gestalts: lg.project_gradient()
    def get_average_blocked_ratio(self):
        ratios = [lg.last_blocked_ratio for lg in self.layer_gestalts if getattr(lg, 'last_blocked_ratio', None) is not None]
        return sum(ratios) / len(ratios) if ratios else 0.0
    def teardown(self):
        for h in self.hooks: h.remove()

class IL2AFeatureAugmentor:
    def __init__(self, d_in, device):
        self.mean_accum = torch.zeros(d_in).to(device)
        self.cov_accum = torch.zeros(d_in, d_in).to(device)
        self.n_samples = 0
        self.mu = self.U = self.S = self.signal_mask = None
        self.n_signal_dims = 0
    def accumulate(self, x):
        N = x.size(0)
        if N == 0: return
        batch_mean = x.mean(dim=0)
        n_old = self.n_samples
        # Welford batched mean update
        self.n_samples += N
        delta = batch_mean - self.mean_accum
        self.mean_accum += delta * (N / self.n_samples)
        # Within-batch scatter (centrado por media del batch)
        x_centered = x - batch_mean
        S_batch = torch.mm(x_centered.t(), x_centered)
        # Corrección inter-grupo: captura varianza entre medias de batches
        if n_old > 0:
            inter = (n_old * N / self.n_samples) * torch.outer(delta, delta)
            self.cov_accum += S_batch + inter
        else:
            self.cov_accum += S_batch
    def fit(self):
        if self.n_samples <= 1: return
        cov = self.cov_accum / max(self.n_samples - 1, 1)
        cov_f = cov.float() + 1e-6 * torch.eye(cov.size(0), device=cov.device)
        U_f, S_f, _ = torch.linalg.svd(cov_f)
        self.mu = self.mean_accum.clone()
        self.U = U_f.to(cov.dtype)
        self.S = S_f.to(cov.dtype)
        self.signal_mask = self.S > self.S.mean()
        self.n_signal_dims = self.signal_mask.sum().item()
    def sample_dreams(self, batch_size, temperature=1.0):
        if self.U is None: return None
        U_sig = self.U[:, self.signal_mask]
        S_sig = self.S[self.signal_mask]
        z = torch.randn(batch_size, self.n_signal_dims, device=self.mu.device) * temperature
        return self.mu + torch.mm(z * torch.sqrt(S_sig), U_sig.t())

class TinyReplayBuffer:
    def __init__(self, per_task=20):
        self.per_task = per_task
        self.buffer = []
    def add_task(self, dataset):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for idx in indices[:self.per_task]:
            item = dataset[idx]
            self.buffer.append({k: item[k] for k in ['input_ids', 'attention_mask', 'label']})
    def sample(self, bs):
        if not self.buffer: return None
        ix = random.choices(range(len(self.buffer)), k=min(bs, len(self.buffer)))
        return {k: torch.stack([self.buffer[i][k] for i in ix]) for k in ['input_ids', 'attention_mask', 'label']}

def calibrate_lora_gestalt(model, dataloader, gestalts):
    """Calibra la Gestalt de LoRA con Hebbian Decay tras cada tarea."""
    hooks = []
    for g in gestalts.values():
        if g.cov_accum is not None:
            g.cov_accum *= HEBBIAN_DECAY
            g.n_samples = int(g.n_samples * HEBBIAN_DECAY)
        def make_hook(gr):
            def hook_fn(mod, inp, out): gr.accumulate(inp[0])
            return hook_fn
        hooks.append(g.lora_module.register_forward_hook(make_hook(g)))
    model.eval()
    with torch.no_grad():
        for i, b in enumerate(dataloader):
            if i >= 50: break
            model(b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
    for h in hooks: h.remove()
    for g in gestalts.values(): g.fit(keep_cov=True)

# =============================================================================
# 4. CARGA DE CLINC150
# =============================================================================
def prepare_clinc150(tokenizer):
    print("[INFO] Cargando CLINC150 (plus)...")
    ds = load_dataset('clinc/clinc_oos', 'plus')
    def tokenize(batch): return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    train_datasets, test_datasets = [], []
    for task_id in range(NUM_TASKS):
        c_start, c_end = task_id * 10, (task_id + 1) * 10 - 1
        t = ds['train'].filter(lambda x: c_start <= x['intent'] <= c_end).map(tokenize, batched=True)
        t = t.rename_column('intent', 'label')
        t.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        train_datasets.append(t)
        
        te = ds['test'].filter(lambda x: c_start <= x['intent'] <= c_end).map(tokenize, batched=True)
        te = te.rename_column('intent', 'label')
        te.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_datasets.append(te)
    return train_datasets, test_datasets

def evaluate_accuracy(model, dataloader, class_end=None):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for b in dataloader:
            out = model(b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
            logits = out.logits
            if class_end is not None:
                logits = logits[:, :class_end]
            c += (logits.argmax(-1) == b['label'].to(device)).sum().item()
            t += b['label'].size(0)
    return (c / t) * 100.0 if t > 0 else 0.0

def calc_aia(R, max_t):
    return np.mean([np.mean([R[i][t] for i in range(t+1)]) for t in range(max_t)])

def save_checkpoint(results, variant_name=None, r_matrix=None):
    if not results: return
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/mega_test_final_results.csv", index=False)
    
    if variant_name and r_matrix is not None:
        filename = f"matriz_R_{variant_name.replace(' ', '_')}.csv"
        np.savetxt(f"{RESULTS_DIR}/{filename}", r_matrix, delimiter=",")
    
    drive_path = "/content/drive/MyDrive/ANTENA_CHECKPOINTS"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(drive_path, exist_ok=True)
        df.to_csv(f"{drive_path}/mega_test_final_results.csv", index=False)
        if variant_name and r_matrix is not None:
            np.savetxt(f"{drive_path}/{filename}", r_matrix, delimiter=",")

# =============================================================================
# 5. MOTOR DE ENTRENAMIENTO CLINC150
# =============================================================================
def run_clinc_model(tokenizer, train_datasets, test_datasets, cfg, mode_name):
    tasks_train = [torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in train_datasets]
    tasks_test = [torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in test_datasets]
    lora_lr = cfg.get('lora_lr', LORA_LR)
    score_lr = cfg.get('score_lr', SCORE_LR)
    rho = cfg.get('rho', 0.99)
    lwf_w = cfg.get('lwf', 5.0)
    use_il2a = cfg.get('use_il2a', False)
    il2a_temp = cfg.get('il2a_temp', 1.0)
    replay_sz = cfg.get('replay_sz', 0)
    
    print(f"\n🚀 ENTRENANDO VARIANTE: {mode_name} | rho={rho}, LwF={lwf_w}, use_il2a={use_il2a}, temp={il2a_temp}, Rep={replay_sz}")
    log_summary(f"\n{'='*50}")
    log_summary(f"🚀 {mode_name} | rho={rho} lwf={lwf_w} il2a={use_il2a} rep={replay_sz}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES, pad_token_id=tokenizer.pad_token_id).to(device)
    
    # Inyectar LoRA (adaptación confinada)
    lora_modules, _originals = inject_lora(model, TARGET_LAYERS, rank=RANK, alpha=32)
    model = model.to(device)
    gestalts = {k: LoRAGestalt(lm, name=k, rho=rho) for k, lm in lora_modules.items()}
    
    egpr = EGPRRegulator(sens=0.2, center=1.5, window=10)
    
    model.eval()
    base_ents = []
    with torch.no_grad():
        for b in tasks_train[0]:
            out = model(b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
            p = torch.softmax(out.logits, -1)
            h = -(p * torch.log(p + 1e-10)).sum(-1)
            base_ents.extend(h.tolist() if h.dim() > 0 else [h.item()])
            if len(base_ents) > 100: break
    egpr.calibrate(base_ents)

    R = np.zeros((NUM_TASKS, NUM_TASKS))
    il2a_gestalt = MidLayerGestalt(model)
    il2a_augmentors = {}
    replay_buf = TinyReplayBuffer(per_task=replay_sz) if replay_sz > 0 else None

    # Hook para calibrar la covarianza de la Capa 10 (Mid-Layer) para IL2A
    target_layer_idx = min(10, len(model.model.layers) - 1)
    layer10_hook = None
    if use_il2a:
        layer10_hook = model.model.layers[target_layer_idx].register_forward_hook(
            lambda m, inp, out: il2a_gestalt.accumulate(out[0] if isinstance(out, tuple) else out)
        )

    for t in range(NUM_TASKS):
        print(f"  [Entrenando Tarea {t+1}/{NUM_TASKS}]...", end="", flush=True)
        class_start = t * 10
        class_end = (t + 1) * 10
        loader = tasks_train[t]

        if t > 0:
            model_frozen = copy.deepcopy(model)
            model_frozen.eval()
            for p in model_frozen.parameters(): p.requires_grad_(False)

        lora_params = []
        for g in gestalts.values():
            lora_params.extend([g.lora_module.lora_A.weight, g.lora_module.lora_B.weight])
        score_params = [model.score.weight]
        if model.score.bias is not None:
            score_params.append(model.score.bias)
        opt = optim.AdamW([
            {'params': lora_params, 'lr': lora_lr},
            {'params': score_params, 'lr': score_lr},
        ])

        model.train()
        for ep in range(EPOCHS_PER_TASK):
            for i, b in enumerate(loader):
                opt.zero_grad()
                inputs, mask, labels = b['input_ids'].to(device), b['attention_mask'].to(device), b['label'].to(device)
                outputs = model(inputs, attention_mask=mask)
                active_logits = outputs.logits[:, :class_end]
                loss_task = F.cross_entropy(active_logits, labels)
                
                loss_lwf = torch.tensor(0.0, device=device)
                loss_mem = torch.tensor(0.0, device=device)

                if t > 0:
                    with torch.no_grad(): frozen_logits = model_frozen(inputs, attention_mask=mask).logits
                    loss_lwf = F.kl_div(F.log_softmax(outputs.logits/LWF_T, dim=-1), F.log_softmax(frozen_logits/LWF_T, dim=-1), reduction='batchmean', log_target=True) * (LWF_T * LWF_T)
                    
                    # IL2A FEATURE REPLAY: Distilación de Sueños Latentes inyectados en Capa 10 (forwarded desde Capa 11)
                    if use_il2a and il2a_gestalt.U is not None:
                        dreams = il2a_gestalt.sample_dreams(batch_size=inputs.size(0), seq_len=1, temperature=il2a_temp)
                        loss_mem = compute_il2a_kl_loss(model, model_frozen, dreams, start_layer_idx=target_layer_idx + 1, class_end=class_end)

                    # IL2A REAL: Feature Augmentation plana al nivel de la cabeza clasificadora
                    if use_il2a and il2a_augmentors:
                        d_list, l_list = [], []
                        sampled = random.sample(list(il2a_augmentors.keys()), min(10, len(il2a_augmentors)))
                        for lbl in sampled:
                            dreams = il2a_augmentors[lbl].sample_dreams(batch_size=8, temperature=il2a_temp)
                            if dreams is not None:
                                d_list.append(dreams)
                                l_list.extend([lbl]*8)
                        if d_list:
                            all_dreams = torch.cat(d_list, dim=0).to(dtype=model.score.weight.dtype, device=device)
                            all_labels = torch.tensor(l_list, device=device)
                            loss_il2a = F.cross_entropy(model.score(all_dreams), all_labels)
                            loss_mem = loss_mem + loss_il2a
                            
                    if replay_sz > 0 and replay_buf is not None:
                        rb = replay_buf.sample(inputs.size(0))
                        if rb is not None:
                            r_out = model(rb['input_ids'].to(device), attention_mask=rb['attention_mask'].to(device), labels=rb['label'].to(device))
                            loss_mem = loss_mem + r_out.loss

                    plastic = egpr.compute(outputs.logits.detach())
                    loss_total = loss_task + lwf_w * loss_lwf + lwf_w * loss_mem
                    loss_total.backward()
                    with torch.no_grad():
                        for g in gestalts.values():
                            g.project_lora_a_grad()
                    opt.param_groups[0]['lr'] = lora_lr * plastic
                    opt.param_groups[1]['lr'] = score_lr * plastic
                else:
                    loss_task.backward()

                opt.step()
                if i % 5 == 0:
                    extra = ""
                    if t > 0:
                        loss_ratio = loss_task.item() / (lwf_w * loss_lwf.item() + 1e-10)
                        extra = f" | LwF:{loss_lwf.item():.4f} | P:{plastic:.3f} | Loss_Ratio:{loss_ratio:.4f}"
                    print(f"  [Ep {ep}] Batch {i}/{len(loader)} | Loss Task: {loss_task.item():.4f}{extra}")

        # PHASE 3: Weight Aligning (post-task bias correction)
        if t > 0:
            with torch.no_grad():
                old_c = torch.arange(t * 10)
                new_c = torch.arange(t * 10, (t + 1) * 10)
                norm_old = model.score.weight[old_c].norm(dim=1).mean()
                norm_new = model.score.weight[new_c].norm(dim=1).mean()
                if norm_new > 1e-6:
                    ratio = norm_old / norm_new
                    model.score.weight[new_c] *= ratio

        # Calibrar IL2A Gestalt en Capa 10 tras finalizar la tarea
        if use_il2a:
            model.eval()
            with torch.no_grad():
                for i, b in enumerate(loader):
                    if i > 50: break
                    model(b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
            il2a_gestalt.fit()

        # Calibrar IL2A Feature Augmentor al final de la tarea
        if use_il2a:
            model.eval()
            with torch.no_grad():
                for b in loader:
                    inputs, mask, labels = b['input_ids'].to(device), b['attention_mask'].to(device), b['label'].to(device)
                    h = model(inputs, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                    seq_lens = (torch.ne(inputs, tokenizer.pad_token_id).sum(-1) - 1).clamp(min=0)
                    pooled_h = h[torch.arange(h.size(0), device=device), seq_lens]
                    for idx in range(labels.size(0)):
                        lbl = labels[idx].item()
                        if lbl not in il2a_augmentors: il2a_augmentors[lbl] = IL2AFeatureAugmentor(model.config.hidden_size, device)
                        il2a_augmentors[lbl].accumulate(pooled_h[idx].unsqueeze(0))
            for lbl in range(class_start, class_start + 10):
                if lbl in il2a_augmentors: il2a_augmentors[lbl].fit()

        if replay_sz > 0:
            replay_buf.add_task(loader.dataset)

        calibrate_lora_gestalt(model, loader, gestalts)

        accs = []
        for e in range(t + 1): 
            acc = evaluate_accuracy(model, tasks_test[e], class_end)
            R[e][t] = acc
            accs.append(f"{acc:.0f}%")
            
        print(f" Completado. Accs: {accs}")
        log_summary(f"  T{t+1}/{NUM_TASKS} → {accs}")
        
        # Real-time sync to Google Drive & auto-save matrix R (Protección contra desconexiones)
        np.savetxt(f"{RESULTS_DIR}/matriz_R_{mode_name}.csv", R, delimiter=",", fmt="%.2f")
        if os.path.exists("/content/drive/MyDrive"):
            drive_dir = "/content/drive/MyDrive/ANTENA_MEGA_TEST"
            os.makedirs(drive_dir, exist_ok=True)
            os.system(f"cp -r {RESULTS_DIR}/* {drive_dir}/ 2>/dev/null || true")
            
    if layer10_hook is not None:
        layer10_hook.remove()
    
    # Fusionar LoRA en los pesos base para transferencia limpia a CausalLM
    with torch.no_grad():
        for name, lora_mod in lora_modules.items():
            lora_mod.base.weight.data += (lora_mod.lora_B.weight.data @ lora_mod.lora_A.weight.data) * lora_mod.scale
    remove_lora(model, TARGET_LAYERS, _originals)
    aia = calc_aia(R, NUM_TASKS)
    bwt = np.mean([R[i][NUM_TASKS-1] - R[i][i] for i in range(NUM_TASKS-1)])
    
    print(f"  --> Final AIA: {aia:.2f}% | BWT: {bwt:.2f}%")
    log_summary(f"📊 {mode_name}: AIA={aia:.2f}% | BWT={bwt:.2f}%")
    return model, aia, bwt, R

# =============================================================================
# WORKERS PARA MULTIPROCESSING (CONCURRENCIA EN GPU)
# =============================================================================
def baseline_worker(tokenizer, queue):
    import torch
    import sys
    import os
    import gc
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = open(f"{RESULTS_DIR}/run_Baseline.log", "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(device)
        base_scores = evaluate_intelligence(base_model, tokenizer)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        queue.put(base_scores)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[ERROR in Baseline]: {err_msg}")
        queue.put({"PIQA": 0.0, "Perplexity": 999.99, "error": err_msg})

def run_variant_worker(nombre, cfg, t_tr, t_te, tokenizer, queue, sem):
    with sem:
        import torch
        import sys
        import os
        import gc
        import numpy as np
        import random
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_file = open(f"{RESULTS_DIR}/run_{nombre}.log", "w", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        seed = cfg.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        try:
            trained_model, aia, bwt, R = run_clinc_model(tokenizer, t_tr, t_te, cfg, nombre)
            print(f"\n--- Evaluando cristalización de inteligencia para {nombre} ---")
            scores = evaluate_intelligence(trained_model, tokenizer)
            print(f"🌟 POST-TEST {nombre} -> PIQA: {scores['PIQA']:.2f}% | Perplejidad: {scores['Perplexity']:.4f}")
            
            del trained_model
            gc.collect()
            torch.cuda.empty_cache()
            
            queue.put({
                "Config": nombre,
                "AIA": f"{aia:.2f}%",
                "BWT": f"{bwt:.2f}%",
                "PIQA": scores['PIQA'],
                "PPL": scores['Perplexity'],
                "R": R.tolist()
            })
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"[ERROR in {nombre}]:\n{err_msg}")
            queue.put({"Config": nombre, "error": err_msg})
            sys.exit(1)

# =============================================================================
# 6. FLUJO MAESTRO
# =============================================================================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    import sys
    import time
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("="*70)
    print("ANTENA MEGA-TEST: CLINC150 + PIQA (Inteligencia Zero-Shot) - PARALELO")
    print("="*70)
    
    print("Pre-descargando modelo y tokenizer para evitar condiciones de carrera en caché...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # Descarga e instanciacion ficticia en CPU para llenar cache
    from transformers import AutoConfig
    AutoConfig.from_pretrained(MODEL_NAME)
    AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    
    t_tr, t_te = prepare_clinc150(tokenizer)
    
    try:
        with open(f"{RESULTS_DIR}/champions_config.json", "r") as f: champs = json.load(f)
    except:
        try:
            with open("champions_config.json", "r") as f: champs = json.load(f)
        except:
            try:
                with open("/content/drive/MyDrive/ANTENA_CHECKPOINTS/champions_config.json", "r") as f: champs = json.load(f)
            except:
                print("[ALERTA] champions_config.json no encontrado. Usando defaults teóricos.")
                champs = {
                    "Zero-Exemplar": {"rho": 0.98, "lwf": 7.0, "use_il2a": False, "il2a_temp": 0.0, "replay_sz": 0, "lora_lr": 5e-4, "score_lr": 1e-4},
                    "IL2A": {"rho": 0.98, "lwf": 7.0, "use_il2a": True, "il2a_temp": 1.0, "replay_sz": 0, "lora_lr": 5e-4, "score_lr": 1e-4},
                    "ANTENA_Hybrid": {"rho": 0.99, "lwf": 5.0, "use_il2a": True, "il2a_temp": 1.0, "replay_sz": 0, "lora_lr": 5e-4, "score_lr": 1e-4},
                    "TinyReplay": {"rho": 0.98, "lwf": 7.0, "use_il2a": False, "il2a_temp": 0.0, "replay_sz": 100, "lora_lr": 5e-4, "score_lr": 1e-4}
                }

    completed_variants = []
    final_results = []
    if os.path.exists(f"{RESULTS_DIR}/mega_test_final_results.csv"):
        print("[INFO] Checkpoint encontrado en RESULTS/. Reanudando ejecución...")
        df_prev = pd.read_csv(f"{RESULTS_DIR}/mega_test_final_results.csv")
        completed_variants = df_prev['Config'].tolist()
        final_results = df_prev.to_dict('records')
    
    default_champs = {
        "ANTENA_Base": {"rho": 0.99, "lwf": 5.0, "use_il2a": True, "il2a_temp": 1.0, "replay_sz": 0, "lora_lr": 5e-4, "score_lr": 1e-4},
        "ANTENA_SlowDistill": {"rho": 0.99, "lwf": 7.5, "use_il2a": True, "il2a_temp": 1.0, "replay_sz": 0, "lora_lr": 3e-4, "score_lr": 1e-4},
        "Rep100": {"rho": 0.99, "lwf": 5.0, "use_il2a": False, "il2a_temp": 0.0, "replay_sz": 100, "lora_lr": 5e-4, "score_lr": 1e-4}
    }
    variantes = [
        ("ANTENA_Base", default_champs["ANTENA_Base"]),
        ("ANTENA_SlowDistill", default_champs["ANTENA_SlowDistill"]),
        ("Rep100", default_champs["Rep100"])
    ]
    
    variantes_to_run = []
    for nombre, cfg in variantes:
        if nombre in completed_variants:
            print(f"\n[SKIP] Variante '{nombre}' ya completada en el checkpoint. Saltando...")
            continue
        variantes_to_run.append((nombre, cfg))

    if variantes_to_run:
        print(f"\n--- FASE 2: EJECUCIÓN EN PARALELO DE {len(variantes_to_run)} VARIANTES ---")
        queue = mp.Queue()
        sem = mp.Semaphore(MAX_PARALLEL_WORKERS)
        processes = []
        for nombre, cfg in variantes_to_run:
            cfg['seed'] = 42
            p = mp.Process(target=run_variant_worker, args=(nombre, cfg, t_tr, t_te, tokenizer, queue, sem))
            p.start()
            processes.append((nombre, p))
            print(f"  [PARALLEL] Proceso para '{nombre}' lanzado (log en: {RESULTS_DIR}/run_{nombre}.log).")
            
        # Monitorear activamente los procesos (Watchdog con Dashboard Live)
        active_processes = list(processes)
        last_print_time = 0
        
        def get_last_log_line(nombre):
            log_path = f"{RESULTS_DIR}/run_{nombre}.log"
            if not os.path.exists(log_path): return "No iniciado"
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        line_strip = line.strip()
                        if line_strip: return line_strip
            except: pass
            return "Leyendo..."

        while active_processes:
            current_time = time.time()
            if current_time - last_print_time >= 5.0:
                status_parts = []
                for nombre, _ in processes:
                    last_line = get_last_log_line(nombre)
                    if "Loading weights" in last_line or "it/s" in last_line:
                        last_line = "Cargando pesos..."
                    if len(last_line) > 50:
                        last_line = "..." + last_line[-47:]
                    status_parts.append(f"{nombre}: {last_line}")
                print(f"[LIVE STATUS] " + " | ".join(status_parts))
                last_print_time = current_time
                
            for nombre, p in list(active_processes):
                if not p.is_alive():
                    if p.exitcode != 0:
                        print(f"\n❌ [CRITICAL] El proceso de la variante '{nombre}' falló con código de salida {p.exitcode}.")
                        time.sleep(1.0)
                        # Recoger errores si los hay en la cola
                        while not queue.empty():
                            res = queue.get()
                            if "error" in res:
                                print(f"Traceback del fallo en '{res['Config']}':\n{res['error']}")
                        print("Cancelando los demás procesos activos...")
                        for _, p_act in active_processes:
                            if p_act.is_alive(): p_act.terminate()
                        sys.exit(p.exitcode)
                    else:
                        print(f"  [PARALLEL] Proceso para '{nombre}' finalizado correctamente.")
                    active_processes.remove((nombre, p))
            time.sleep(1.0)
            
        # Recoger resultados
        while not queue.empty():
            res = queue.get()
            if "error" in res:
                continue
            
            final_results.append({
                "Config": res['Config'],
                "CLINC_AIA": res['AIA'],
                "BWT": res['BWT'],
                "PIQA": res['PIQA'],
                "PPL": res['PPL']
            })
            R_mat = np.array(res['R'])
            save_checkpoint(final_results, variant_name=res['Config'], r_matrix=R_mat)

    print("\n=======================================================")
    print("🏆 RESULTADOS FINALES DE CRISTALIZACIÓN (PAPER)")
    print("=======================================================")
    df = pd.DataFrame(final_results)
    print(df.to_string(index=False))
    
    import os
    os.system(f"zip -r final_mega_test_output.zip {RESULTS_DIR}/")
