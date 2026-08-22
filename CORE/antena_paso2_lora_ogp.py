#!/usr/bin/env python3
"""
ANTENA Gestalt — Paso 2: LoRA Continuo + Soft-OGP
===================================================
Valida la Inmunidad Topológica en SmolLM-135M mediante LoRA fine-tuning
con proyección ortogonal de gradientes.

Correcciones Matemáticas Aplicadas:
  #1: Soft-OGP se aplica SOLO a lora_A.weight.grad (Equipo ANTENA_NEURAL)
  #4: Piso de exploración 0.02 en EGPR (ya en antena_gestalt_core.py)
  #5: Gradientes in-place con torch.no_grad() (ya en antena_gestalt_core.py)

Experimento A/B:
  A) LoRA SIN protección   → medir olvido (perplexity T1 sube)
  B) LoRA CON Soft-OGP     → medir inmunidad (perplexity T1 estable)

Uso:
  python3 antena_paso2_lora_ogp.py
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from antena_gestalt_core import EGPRRegulator

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
TARGET_LAYERS = [14, 19, 24, 29]  # Últimas capas (razonamiento semántico)
TRAIN_STEPS = 30
BASE_LR = 5e-4
RHO_OGP = 0.8  # Factor de atenuación Soft-OGP

# =============================================================================
# DATOS EMBEBIDOS
# =============================================================================

# Tarea 1: Conocimiento general (lo que SmolLM ya sabe bien)
TASK1_CALIBRATION = (
    "The water cycle describes how water evaporates from the surface of the earth, "
    "rises into the atmosphere, cools and condenses into rain or snow in clouds, "
    "and falls again to the surface as precipitation. Rivers flow from mountains "
    "to the sea, carrying minerals and nutrients along the way. Forests play a "
    "crucial role in maintaining the balance of carbon dioxide and oxygen. "
    "The rotation of the Earth causes day and night, while its orbit around "
    "the Sun creates the four seasons that define our calendar year."
)

TASK1_VALIDATION = (
    "The history of mathematics begins with the ancient civilizations of "
    "Mesopotamia and Egypt. These early cultures developed systems of counting "
    "and measuring that were essential for agriculture, trade, and construction. "
    "The Greeks later formalized mathematical reasoning and developed geometry. "
    "Mathematics continued to evolve through the contributions of many scholars."
)

# Tarea 2: Dominio especializado (bioquímica densa)
TASK2_TRAINING = (
    "The ribosome facilitates peptide bond formation between amino acids during "
    "translation. Transfer RNA molecules carry specific amino acids to the ribosome "
    "where the anticodon pairs with the messenger RNA codon. Post-translational "
    "modifications include phosphorylation, glycosylation, and ubiquitination. "
    "The endoplasmic reticulum processes newly synthesized proteins through folding "
    "and quality control mechanisms. Misfolded proteins are targeted for degradation "
    "by the proteasome complex via the ubiquitin-dependent pathway. Chaperone proteins "
    "assist in the proper folding of polypeptide chains within the cellular environment."
)

# =============================================================================
# 1. CUSTOM LoRA (sin dependencia de peft)
# =============================================================================

class LoRALinear(nn.Module):
    """
    Adaptador LoRA mínimo: W_eff = W0 + (B @ A) * scale
    - W0 (base): congelado
    - A: (rank, d_in)  → opera en espacio de entrada (donde vive la Gestalt)
    - B: (d_out, rank)  → opera en espacio intermedio
    - B se inicializa a cero → output inicial = W0 (sin perturbación)
    """
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base_layer.weight.shape
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Linear(d_in, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, d_out, bias=False, dtype=dtype)
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


# =============================================================================
# 2. LoRA GESTALT (Soft-OGP específico para lora_A — Corrección #1)
# =============================================================================

class LoRAGestalt:
    """
    Veto Topológico adaptado para LoRA.
    
    Corrección Matemática #1 (Equipo ANTENA_NEURAL):
      La Gestalt se calcula sobre el espacio de ENTRADA (d_in).
      lora_A mapea d_in → rank, así que su gradiente vive en d_in.
      lora_B mapea rank → d_out (espacio intermedio distinto).
      → Soft-OGP se aplica SOLO a lora_A.weight.grad.
    """
    def __init__(self, lora_module, name="", rho=0.8):
        self.lora_module = lora_module
        self.name = name
        self.rho = rho
        self.U = None
        self.S = None
        self.signal_mask = None
        self.cov_accum = None
        self.mean_accum = None
        self.n_samples = 0

    def accumulate(self, x, mask=None):
        """Acumula covarianza con Welford Batched (corrige sesgo inter-batch)."""
        with torch.no_grad():
            if mask is not None and x.dim() == 3:
                x_flat = x[mask.bool()]
            else:
                x_flat = x.reshape(-1, x.size(-1))
            N = x_flat.size(0)
            if N == 0:
                return
            batch_mean = x_flat.mean(dim=0)
            x_centered = x_flat - batch_mean
            S_batch = torch.mm(x_centered.t(), x_centered)
            
            if self.mean_accum is None:
                self.mean_accum = batch_mean.clone()
                self.cov_accum = S_batch
                self.n_samples = N
            else:
                n_old = self.n_samples
                delta = batch_mean - self.mean_accum
                n_new = n_old + N
                inter = (n_old * N / n_new) * torch.outer(delta, delta)
                self.mean_accum = self.mean_accum + delta * (N / n_new)
                self.cov_accum = self.cov_accum + S_batch + inter
                self.n_samples = n_new

    def fit(self, keep_cov=False):
        """SVD + Heurística de Ledoit-Péché para extraer la Gestalt."""
        if self.n_samples == 0:
            return
        cov = self.cov_accum / (self.n_samples - 1)
        # Estabilización numérica para evitar cuelgues en LAPACK con matrices de rango deficiente
        cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device)
        U, S, _ = torch.linalg.svd(cov)
        self.U = U
        self.S = S
        self.signal_mask = S > S.mean()
        n_protected = self.signal_mask.sum().item()
        print(f"    [Gestalt] {self.name}: {n_protected}/{len(S)} dims protegidas")
        
        if not keep_cov:
            self.cov_accum = None
            self.mean_accum = None
            self.n_samples = 0

    def project_lora_a_grad(self):
        """Proyecta SOLO el gradiente de lora_A ortogonalmente a la Gestalt."""
        grad = self.lora_module.lora_A.weight.grad
        if grad is None:
            return

        # Si tenemos proyecciones mezcladas de múltiples dominios
        blend_projs = getattr(self, "blend_projections", None)
        if blend_projs:
            current_grad = grad.clone()
            for U_d, mask_d in blend_projs:
                if U_d is None or mask_d is None:
                    continue
                attenuation = torch.ones(U_d.size(1), dtype=current_grad.dtype, device=current_grad.device)
                attenuation[mask_d] = 1.0 - self.rho
                
                U_d_cast = U_d.to(device=current_grad.device, dtype=current_grad.dtype)
                g_rot = torch.mm(current_grad, U_d_cast)
                g_rot = g_rot * attenuation
                current_grad = torch.mm(g_rot, U_d_cast.t())
            self.lora_module.lora_A.weight.grad.copy_(current_grad)
        else:
            if self.U is None or self.signal_mask is None:
                return
            # grad shape: (rank, d_in) — cada fila es un vector en espacio d_in
            attenuation = torch.ones_like(self.S, dtype=grad.dtype)
            attenuation[self.signal_mask] = 1.0 - self.rho
            
            U_dtype = self.U.to(grad.dtype)

            g_rot = torch.mm(grad, U_dtype)       # Rotar al espacio Gestalt
            g_rot = g_rot * attenuation           # Atenuar dimensiones protegidas
            g_proj = torch.mm(g_rot, U_dtype.t())  # Rotar de vuelta

            self.lora_module.lora_A.weight.grad.copy_(g_proj)


# =============================================================================
# 3. INYECCIÓN Y LIMPIEZA DE LoRA
# =============================================================================

def inject_lora(model, target_layers, rank=8, alpha=16):
    """Reemplaza q_proj y v_proj de las capas objetivo con LoRALinear."""
    lora_modules = {}
    originals = {}
    for i in target_layers:
        attn = model.model.layers[i].self_attn
        originals[f"L{i}_q"] = attn.q_proj
        originals[f"L{i}_v"] = attn.v_proj
        lora_q = LoRALinear(attn.q_proj, rank=rank, alpha=alpha)
        lora_v = LoRALinear(attn.v_proj, rank=rank, alpha=alpha)
        attn.q_proj = lora_q
        attn.v_proj = lora_v
        lora_modules[f"L{i}_q"] = lora_q
        lora_modules[f"L{i}_v"] = lora_v
    return lora_modules, originals


def remove_lora(model, target_layers, originals):
    """Restaura las capas originales."""
    for i in target_layers:
        attn = model.model.layers[i].self_attn
        attn.q_proj = originals[f"L{i}_q"]
        attn.v_proj = originals[f"L{i}_v"]


# =============================================================================
# 4. PERPLEXITY
# =============================================================================

def compute_perplexity(model, tokenizer, text):
    """Calcula la perplejidad del modelo sobre un texto dado."""
    model.eval()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=tokens, labels=tokens)
    return torch.exp(outputs.loss).item()


# =============================================================================
# 5. CALIBRACIÓN DE GESTALT
# =============================================================================

def calibrate_gestalt(model, tokenizer, gestalts, text, n_passes=5):
    """
    Pasa texto base por el modelo para acumular covarianza de entradas
    en cada módulo LoRA, luego ejecuta SVD para sellar la Gestalt.
    """
    print("  [Calibración] Acumulando covarianza del espacio de entrada...")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    hooks = []

    # Instalar hooks de captura
    for key, g in gestalts.items():
        lora_mod = g.lora_module
        def make_hook(gestalt_ref):
            def hook_fn(mod, inp, out):
                gestalt_ref.accumulate(inp[0])
            return hook_fn
        hooks.append(lora_mod.register_forward_hook(make_hook(g)))

    model.eval()
    with torch.no_grad():
        for _ in range(n_passes):
            model(input_ids=tokens)

    # Limpiar hooks
    for h in hooks:
        h.remove()

    # Sellar la Gestalt (SVD)
    print("  [Calibración] Ejecutando SVD + Ledoit-Péché...")
    for g in gestalts.values():
        g.fit()


# =============================================================================
# 6. ENTRENAMIENTO LoRA
# =============================================================================

def train_lora(model, tokenizer, gestalts, text, steps, lr, use_ogp=False):
    """
    Entrena los adaptadores LoRA sobre el texto de Tarea 2.
    Si use_ogp=True, aplica Soft-OGP sobre lora_A.weight.grad tras backward().
    """
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]

    # Recoger solo los parámetros LoRA para el optimizador
    lora_params = []
    for g in gestalts.values():
        lm = g.lora_module
        lora_params.extend([lm.lora_A.weight, lm.lora_B.weight])
    optimizer = torch.optim.Adam(lora_params, lr=lr)

    # EGPR para modulación de LR
    egpr = EGPRRegulator(sens=0.1, center=2.0, window=30)

    model.train()
    tag = "CON Soft-OGP" if use_ogp else "SIN protección"
    print(f"\n  [Training {tag}] {steps} pasos, lr={lr}")

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(input_ids=tokens, labels=tokens)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()

        # --- Veto Termodinámico (EGPR) ---
        plasticity = egpr.compute(logits[:, -1, :])
        for pg in optimizer.param_groups:
            pg['lr'] = lr * plasticity

        # --- Veto Topológico (Soft-OGP) sobre lora_A únicamente ---
        if use_ogp:
            for g in gestalts.values():
                g.project_lora_a_grad()

        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            print(f"    Paso {step:3d}/{steps} | Loss: {loss.item():.4f} | Plast: {plasticity:.4f}")

    model.eval()


# =============================================================================
# 7. EXPERIMENTO PRINCIPAL (A/B)
# =============================================================================

def run_experiment(use_ogp, model, tokenizer):
    """Ejecuta un experimento completo: inject → calibrate → train → evaluate."""
    tag = "CON Soft-OGP" if use_ogp else "SIN protección"
    print(f"\n{'='*60}")
    print(f" EXPERIMENTO: LoRA {tag}")
    print(f"{'='*60}")

    # 1. Perplexity base (antes de tocar nada)
    ppl_t1_before = compute_perplexity(model, tokenizer, TASK1_VALIDATION)
    ppl_t2_before = compute_perplexity(model, tokenizer, TASK2_TRAINING)
    print(f"  [Baseline] PPL Tarea1: {ppl_t1_before:.2f} | PPL Tarea2: {ppl_t2_before:.2f}")

    # 2. Inyectar LoRA
    lora_modules, originals = inject_lora(model, TARGET_LAYERS, LORA_RANK, LORA_ALPHA)
    print(f"  [LoRA] Inyectados {len(lora_modules)} adaptadores (rank={LORA_RANK})")

    # 3. Crear LoRAGestalt para cada módulo
    gestalts = {}
    for key, lm in lora_modules.items():
        gestalts[key] = LoRAGestalt(lm, name=key, rho=RHO_OGP)

    # 4. Calibrar Gestalt (acumular covarianza sobre Tarea 1)
    if use_ogp:
        calibrate_gestalt(model, tokenizer, gestalts, TASK1_CALIBRATION, n_passes=5)
    else:
        print("  [Calibración] Saltada (sin protección)")

    # 5. Entrenar LoRA sobre Tarea 2
    train_lora(model, tokenizer, gestalts, TASK2_TRAINING,
               steps=TRAIN_STEPS, lr=BASE_LR, use_ogp=use_ogp)

    # 6. Evaluar post-entrenamiento
    ppl_t1_after = compute_perplexity(model, tokenizer, TASK1_VALIDATION)
    ppl_t2_after = compute_perplexity(model, tokenizer, TASK2_TRAINING)

    delta_t1 = ppl_t1_after - ppl_t1_before
    delta_t2 = ppl_t2_after - ppl_t2_before

    print(f"\n  ┌{'─'*56}┐")
    print(f"  │ RESULTADOS: LoRA {tag:20s}                   │")
    print(f"  ├{'─'*56}┤")
    print(f"  │ PPL Tarea1 (Retención):  {ppl_t1_before:8.2f} → {ppl_t1_after:8.2f}  (Δ={delta_t1:+.2f}) │")
    print(f"  │ PPL Tarea2 (Adquisición): {ppl_t2_before:8.2f} → {ppl_t2_after:8.2f}  (Δ={delta_t2:+.2f}) │")
    print(f"  └{'─'*56}┘")

    # 7. Restaurar modelo original
    remove_lora(model, TARGET_LAYERS, originals)

    return {
        "tag": tag,
        "ppl_t1_before": ppl_t1_before, "ppl_t1_after": ppl_t1_after,
        "ppl_t2_before": ppl_t2_before, "ppl_t2_after": ppl_t2_after,
        "delta_t1": delta_t1, "delta_t2": delta_t2,
    }


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║  ANTENA Gestalt — Paso 2: LoRA + Soft-OGP (A/B Test)    ║")
    print("╚" + "═" * 58 + "╝")

    # Cargar modelo
    print("\n[CARGA] Descargando/cargando SmolLM-135M-Instruct...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.to("cpu")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[CARGA] Listo en {time.time()-t0:.1f}s")

    # Congelar todo el modelo base
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Experimento A: SIN protección ---
    results_a = run_experiment(use_ogp=False, model=model, tokenizer=tokenizer)

    # --- Experimento B: CON Soft-OGP ---
    results_b = run_experiment(use_ogp=True, model=model, tokenizer=tokenizer)

    # --- Tabla comparativa final ---
    print("\n" + "=" * 60)
    print(" COMPARATIVA FINAL: Inmunidad Topológica")
    print("=" * 60)
    print(f"  {'Métrica':<30s} | {'SIN OGP':>10s} | {'CON OGP':>10s}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'PPL T1 antes':<30s} | {results_a['ppl_t1_before']:>10.2f} | {results_b['ppl_t1_before']:>10.2f}")
    print(f"  {'PPL T1 después':<30s} | {results_a['ppl_t1_after']:>10.2f} | {results_b['ppl_t1_after']:>10.2f}")
    print(f"  {'Δ PPL T1 (olvido)':<30s} | {results_a['delta_t1']:>+10.2f} | {results_b['delta_t1']:>+10.2f}")
    print(f"  {'PPL T2 después':<30s} | {results_a['ppl_t2_after']:>10.2f} | {results_b['ppl_t2_after']:>10.2f}")
    print(f"  {'Δ PPL T2 (aprendizaje)':<30s} | {results_a['delta_t2']:>+10.2f} | {results_b['delta_t2']:>+10.2f}")
    print("=" * 60)

    # Veredicto
    if abs(results_b['delta_t1']) < abs(results_a['delta_t1']):
        ratio = abs(results_a['delta_t1']) / max(abs(results_b['delta_t1']), 0.01)
        print(f"\n  ✅ Soft-OGP reduce el olvido en Tarea 1 por un factor de {ratio:.1f}x")
    else:
        print(f"\n  ⚠ Resultado inesperado — requiere investigación adicional")

    print("\nPaso 2 completado.")


if __name__ == "__main__":
    main()
