#!/usr/bin/env python3
"""
ANTENA SLM — General Ability Delta (GAD) Benchmark
=======================================================
Measures zero-shot PIQA accuracy before and after disruptive fine-tuning.
Compares: Vanilla | EWC | ANTENA Fase 6
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import sys
import copy
import json
import time

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
from antena_gestalt_core import EGPRRegulator
from antena_paso2_lora_ogp import LoRAGestalt, inject_lora
from antena_latent_il2a import MidLayerGestalt, compute_il2a_kl_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
MAX_LENGTH = 128
BATCH_SIZE = 8
SAMPLES_PER_TASK = 500
EPOCHS_PER_TASK = 3
BASE_LR = 5e-4
KL_WEIGHT = 0.5
TARGET_LAYERS = list(range(10, 30))
RANK = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# PIQA EVALUATION
# ============================================================
def load_piqa_validation():
    import urllib.request
    if not os.path.exists("valid.jsonl"):
        urllib.request.urlretrieve("https://yonatanbisk.com/piqa/data/valid.jsonl", "valid.jsonl")
        urllib.request.urlretrieve("https://yonatanbisk.com/piqa/data/valid-labels.lst", "valid-labels.lst")

    data = []
    with open("valid.jsonl", "r") as f_data, open("valid-labels.lst", "r") as f_labels:
        for line_data, line_label in zip(f_data, f_labels):
            item = json.loads(line_data)
            item['label'] = int(line_label.strip())
            data.append(item)
    return data


def evaluate_piqa_zeroshot(model, tokenizer, n_samples=500):
    print(f"  -> Evaluating PIQA (Zero-Shot, n={n_samples})...")
    ds = load_piqa_validation()[:n_samples]

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in ds:
            goal = item['goal']
            s1, s2 = item['sol1'], item['sol2']
            label = item['label']

            prompt1 = f"Goal: {goal} Solution: {s1}"
            prompt2 = f"Goal: {goal} Solution: {s2}"

            inputs1 = tokenizer(prompt1, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
            inputs2 = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

            loss1 = model(inputs1.input_ids, labels=inputs1.input_ids).loss.item()
            loss2 = model(inputs2.input_ids, labels=inputs2.input_ids).loss.item()

            pred = 0 if loss1 < loss2 else 1
            if pred == label:
                correct += 1
            total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"  -> PIQA Accuracy: {acc:.2f}%")
    return acc


# ============================================================
# DISRUPTION TRAINING
# ============================================================
def load_disruption_data(tokenizer):
    print("  -> Loading disruption task (AG News)...")
    ds = load_dataset("fancyzhx/ag_news", split="train", trust_remote_code=True)
    ds = ds.select(range(min(SAMPLES_PER_TASK, len(ds))))

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True), ds


# ============================================================
# EWC for LM
# ============================================================
class EWC_LM:
    def __init__(self, model, lam=100.0):
        self.lam = lam
        self.params_old = {}
        self.fisher = {}

    def compute_fisher(self, model, dataloader, num_batches=30):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        count = 0
        for batch in dataloader:
            if count >= num_batches:
                break
            model.zero_grad()
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            outputs = model(inputs, attention_mask=mask, labels=inputs)
            outputs.loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
            count += 1
        for n in fisher:
            fisher[n] /= count
        self.fisher = fisher
        self.params_old = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params_old[n]).pow(2)).sum()
        return self.lam * loss


# ============================================================
# EXPERIMENT RUNNERS
# ============================================================

def run_gad_vanilla(tokenizer, disruption_loader):
    print(f"\n{'='*60}")
    print(f"GAD: VANILLA")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    acc_pre = evaluate_piqa_zeroshot(model, tokenizer)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(EPOCHS_PER_TASK):
        for i, batch in enumerate(disruption_loader):
            optimizer.zero_grad()
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['input_ids'].to(device))
            outputs.loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f"  Batch {i} | Loss: {outputs.loss.item():.4f}")

    acc_post = evaluate_piqa_zeroshot(model, tokenizer)
    return acc_pre, acc_post


def run_gad_ewc(tokenizer, disruption_loader, calibration_loader):
    print(f"\n{'='*60}")
    print(f"GAD: EWC (λ=100)")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    acc_pre = evaluate_piqa_zeroshot(model, tokenizer)

    # Compute Fisher on calibration data (wiki or similar)
    ewc = EWC_LM(model, lam=100.0)
    ewc.compute_fisher(model, calibration_loader)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(EPOCHS_PER_TASK):
        for i, batch in enumerate(disruption_loader):
            optimizer.zero_grad()
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['input_ids'].to(device))
            loss = outputs.loss + ewc.penalty(model)
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f"  Batch {i} | Loss: {outputs.loss.item():.4f}")

    acc_post = evaluate_piqa_zeroshot(model, tokenizer)
    return acc_pre, acc_post


def run_gad_antena(tokenizer, disruption_loader):
    print(f"\n{'='*60}")
    print(f"GAD: ANTENA FASE 6")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # Inject LoRA
    lora_modules, _ = inject_lora(model, TARGET_LAYERS, rank=RANK, alpha=32)
    model = model.to(device)
    gestalts = {k: LoRAGestalt(lm, name=k, rho=0.9) for k, lm in lora_modules.items()}
    mid_gestalt = MidLayerGestalt(model)
    egpr = EGPRRegulator(sens=0.2, center=3.0, window=10)

    acc_pre = evaluate_piqa_zeroshot(model, tokenizer)

    # Calibrate on disruption data (pre-task)
    print("  [Gestalt] Calibrating...")
    hooks = []
    for key, g in gestalts.items():
        def make_hook(gestalt_ref):
            def hook_fn(mod, inp, out):
                gestalt_ref.accumulate(inp[0])
            return hook_fn
        hooks.append(g.lora_module.register_forward_hook(make_hook(g)))

    target_layer_idx = TARGET_LAYERS[0] - 1
    mid_hook = model.model.layers[target_layer_idx].register_forward_hook(
        lambda m, inp, out: mid_gestalt.accumulate(out[0])
    )
    hooks.append(mid_hook)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(disruption_loader):
            if i >= 50:
                break
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            model(input_ids=inputs, attention_mask=mask)
    for h in hooks:
        h.remove()

    for g in gestalts.values():
        g.fit(keep_cov=True)
    mid_gestalt.fit(keep_cov=True)

    # EGPR calibration
    base_ents = []
    with torch.no_grad():
        for i, batch in enumerate(disruption_loader):
            if i >= 20:
                break
            inputs = batch['input_ids'].to(device)
            out = model(inputs)
            p = torch.softmax(out.logits[:, -1, :], -1)
            h = -(p * torch.log(p + 1e-10)).sum(-1)
            base_ents.extend(h.tolist())
    egpr.calibrate(base_ents)

    # Frozen model + dreams
    model_frozen = copy.deepcopy(model)
    model_frozen.eval()
    dreams = mid_gestalt.sample_dreams(batch_size=BATCH_SIZE).to(device)

    # Train
    lora_params = []
    for g in gestalts.values():
        lm = g.lora_module
        lora_params.extend([lm.lora_A.weight, lm.lora_B.weight])
    optimizer = optim.AdamW(lora_params, lr=BASE_LR)

    model.train()
    for epoch in range(EPOCHS_PER_TASK):
        for i, batch in enumerate(disruption_loader):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            outputs = model(inputs, attention_mask=mask, labels=inputs)
            loss_task = outputs.loss

            with torch.no_grad():
                logits_last = outputs.logits[:, -1, :]
            plasticity = egpr.compute(logits_last)

            loss_kl = compute_il2a_kl_loss(model, model_frozen, dreams, start_layer_idx=TARGET_LAYERS[0]-1, T=2.0)
            loss_total = loss_task + (KL_WEIGHT * loss_kl)
            loss_total.backward()

            for g in gestalts.values():
                g.project_lora_a_grad()

            for pg in optimizer.param_groups:
                pg['lr'] = BASE_LR * plasticity

            optimizer.step()
            if i % 20 == 0:
                print(f"  Batch {i} | Loss: {loss_task.item():.4f} | KL: {loss_kl.item():.4f}")

    acc_post = evaluate_piqa_zeroshot(model, tokenizer)
    return acc_pre, acc_post


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("ANTENA SLM — GAD (General Ability Delta) Benchmark Suite")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    disruption_loader, _ = load_disruption_data(tokenizer)

    # For EWC: need a calibration loader (use wikitext)
    from datasets import load_dataset as ld
    wiki = ld('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True)
    wiki = wiki.filter(lambda x: len(x['text'].strip()) > 50).select(range(250))

    def tok_wiki(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    wiki = wiki.map(tok_wiki, batched=True, remove_columns=wiki.column_names)
    wiki.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    wiki_loader = torch.utils.data.DataLoader(wiki, batch_size=BATCH_SIZE)

    # Run experiments
    van_pre, van_post = run_gad_vanilla(tokenizer, disruption_loader)
    ewc_pre, ewc_post = run_gad_ewc(tokenizer, disruption_loader, wiki_loader)
    ant_pre, ant_post = run_gad_antena(tokenizer, disruption_loader)

    # Results
    print("\n" + "=" * 60)
    print(" GAD RESULTS (PIQA Zero-Shot Accuracy)")
    print("=" * 60)
    results = []
    for name, pre, post in [("Vanilla", van_pre, van_post), ("EWC (λ=100)", ewc_pre, ewc_post), ("ANTENA Fase 6", ant_pre, ant_post)]:
        delta = post - pre
        print(f"  {name:20s} | Pre: {pre:.1f}% | Post: {post:.1f}% | GAD: {delta:+.1f}%")
        results.append({'Method': name, 'PIQA_Pre': pre, 'PIQA_Post': post, 'GAD': delta})

    df = pd.DataFrame(results)
    os.makedirs("RESULTS", exist_ok=True)
    df.to_csv("RESULTS/slm_gad_benchmark.csv", index=False)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    import zipfile
    import glob
    with zipfile.ZipFile('RESULTS/telemetria_gad.zip', 'w') as zf:
        zf.write('RESULTS/slm_gad_benchmark.csv')

    # Download handled by notebook's final cell to avoid duplicates
