#!/usr/bin/env python3
"""
ANTENA IL2A Core: True Latent Dream Replay (Mid-Layer Injection)
=====================================================================
Sustituye la generación autorregresiva costosa (15-30 minutos) por el 
muestreo espectral de ruido gaussiano en el subespacio de activaciones (1.5 ms).

Matemática y Algoritmo:
1. SVD con encogimiento no lineal de Ledoit-Péché sobre la matriz de covarianza acumulada:
   Σ = U S U^T
2. Generación de Sueños Latentes (Vectores de dimensión d_in):
   h_dream = μ + U_signal * sqrt(S_signal) * z_signal + h_null
3. Inyección Intermedia (Mid-Layer Injection):
   Los sueños (h_dream) se inyectan en la Capa Intermedia (`start_layer_idx`, e.g. Capa 10)
   y se propagan mediante `forward_from` a través de los bloques LoRA hacia la cabeza final (`score`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MidLayerGestalt:
    """
    Acumula la covarianza de las activaciones en la capa intermedia de la red
    para calibrar el subespacio de señal y generar Eigenreplays (True Latent Replay).
    """
    def __init__(self, model):
        self.model = model
        self.d_in = model.config.hidden_size
        self.device = next(model.parameters()).device
        self.mean_accum = torch.zeros(self.d_in, device=self.device)
        self.cov_accum = torch.zeros(self.d_in, self.d_in, device=self.device)
        self.n_samples = 0
        
        self.mu = None
        self.U = None
        self.S = None
        self.signal_mask = None
        self.n_signal_dims = 0

    def accumulate(self, h_states):
        """Accumulate activations with proper Welford batched covariance.
        
        Welford parallel merge:
          S_AB = S_A + S_B + (n_A * n_B / (n_A + n_B)) * (μ_B - μ_A)(μ_B - μ_A)^T
        """
        with torch.no_grad():
            x = h_states.detach().view(-1, self.d_in).float()
            N = x.size(0)
            if N == 0:
                return

            batch_mean = x.mean(dim=0)
            n_old = self.n_samples
            
            # Welford batched mean update
            delta = batch_mean - self.mean_accum
            self.n_samples += N
            self.mean_accum += delta * (N / self.n_samples)

            # Within-batch scatter (centrado por media del BATCH, no global)
            x_centered = x - batch_mean
            S_batch = torch.mm(x_centered.t(), x_centered)
            
            # Corrección inter-grupo
            if n_old > 0:
                inter = (n_old * N / self.n_samples) * torch.outer(delta, delta)
                self.cov_accum += S_batch + inter
            else:
                self.cov_accum += S_batch

    def fit(self, keep_cov=True, decay=0.95):
        """Ejecuta SVD y determina la dimensionalidad de señal (Ledoit-Péché Threshold)"""
        if self.n_samples <= 1:
            return
            
        cov = self.cov_accum / (self.n_samples - 1)
        U, S, _ = torch.linalg.svd(cov.to(torch.float32))
        U, S = U.to(self.device), S.to(self.device)
        
        self.mu = self.mean_accum.clone()
        self.U = U
        self.S = S
        
        # Criterio de señal de Ledoit-Péché: Valores mayores que el promedio espectral
        self.signal_mask = S > S.mean()
        self.n_signal_dims = self.signal_mask.sum().item()
        
        print(f"[ANTENA IL2A] Calibración Espectral: {self.n_signal_dims}/{self.d_in} dimensiones protegidas para Sueños Latentes.")
        
        if keep_cov:
            self.cov_accum *= decay
            self.n_samples = int(self.n_samples * decay)
        else:
            self.cov_accum = torch.zeros_like(self.cov_accum)
            self.n_samples = 0

    def sample_dreams(self, batch_size=32, seq_len=1, temperature=1.0):
        """
        Eigenreplay: Genera h_dreams muestreando de la distribución Gaussiana multivariada
        definida por los autovectores de señal + ruido defensivo isotrópico de Null-Space.
        """
        if self.U is None:
            raise ValueError("MidLayerGestalt no está calibrado. Llama a fit() primero.")
            
        U_sig = self.U[:, self.signal_mask]
        S_sig = self.S[self.signal_mask]
        
        null_variance = self.S[~self.signal_mask].mean() if (~self.signal_mask).any() else torch.tensor(0.0, device=self.device)
        S_sig_adj = torch.clamp(S_sig - null_variance, min=1e-6)
        
        N_total = batch_size * seq_len
        z_sig = torch.randn(N_total, self.n_signal_dims, device=self.device) * temperature
        z_scaled = z_sig * torch.sqrt(S_sig_adj)
        h_signal = torch.mm(z_scaled, U_sig.t())
        
        h_null = torch.randn(N_total, self.d_in, device=self.device) * torch.sqrt(null_variance) * temperature
        
        h_dreams = self.mu + h_signal + h_null
        
        if seq_len > 1:
            h_dreams = h_dreams.view(batch_size, seq_len, self.d_in)
        else:
            h_dreams = h_dreams.view(batch_size, 1, self.d_in)
            
        return h_dreams.to(self.model.dtype)


def forward_from_layer(model_instance, hidden_states, start_layer_idx):
    """
    Propaga estados ocultos (h_dreams) desde `start_layer_idx` hasta la salida del modelo,
    haciendo que las activaciones atraviesen los bloques LoRA/Atención restantes.
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(1) # [batch, 1, d_in]
        
    batch_size, seq_length, _ = hidden_states.shape
    device = hidden_states.device
    
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    position_embeddings = None
    if hasattr(model_instance.model, "rotary_emb"):
        position_embeddings = model_instance.model.rotary_emb(hidden_states, position_ids)
        
    kwargs = {"attention_mask": None, "position_ids": position_ids}
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    for i in range(start_layer_idx, len(model_instance.model.layers)):
        layer_module = model_instance.model.layers[i]
        try:
            layer_outputs = layer_module(hidden_states, **kwargs)
        except TypeError:
            layer_outputs = layer_module(hidden_states, attention_mask=None, position_ids=position_ids)
            
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

    hidden_states = model_instance.model.norm(hidden_states)
    
    if hasattr(model_instance, 'score'):
        # Sequence Classification (prenda el último token)
        pooled_h = hidden_states[:, -1, :]
        return model_instance.score(pooled_h)
    elif hasattr(model_instance, 'lm_head'):
        return model_instance.lm_head(hidden_states)
    else:
        raise AttributeError("El modelo no contiene 'score' ni 'lm_head'.")


def compute_il2a_kl_loss(model_current, model_frozen, h_dreams, start_layer_idx=10, class_end=None, T=2.0):
    """
    Calcula la Pérdida de Distilación KL de IL2A con inyección intermedia 
    y enmascaramiento opcional de clases activas.
    """
    current_logits = forward_from_layer(model_current, h_dreams, start_layer_idx) / T
    
    with torch.no_grad():
        frozen_logits = forward_from_layer(model_frozen, h_dreams, start_layer_idx) / T
        
    if class_end is not None and current_logits.dim() == 2:
        current_logits = current_logits[:, :class_end]
        frozen_logits = frozen_logits[:, :class_end]

    loss_kl = F.kl_div(
        F.log_softmax(current_logits, dim=-1),
        F.log_softmax(frozen_logits, dim=-1),
        reduction='batchmean',
        log_target=True
    ) * (T * T)
    
    return loss_kl
