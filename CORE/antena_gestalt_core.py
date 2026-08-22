import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EGPRRegulator:
    """Veto Termodinámico (Global Entropy Gate)"""
    def __init__(self, sens=0.1, center=2.0, window=30):
        self.sens = sens; self.center = center; self.window = window
        self.hist = []; self.cal_mu = None; self.cal_std = None

    def calibrate(self, ents):
        self.cal_mu = np.mean(ents)
        self.cal_std = np.std(ents) + 1e-9
        self.hist = []

    def compute(self, logits):
        p = torch.softmax(logits.detach(), -1)
        h = -(p * torch.log(p + 1e-10)).sum(-1).mean().item()
        self.hist.append(h)
        if len(self.hist) > self.window: self.hist.pop(0)
        
        if self.cal_mu is None: return 1.0
        
        z = (h - self.cal_mu) / self.cal_std
        base = max(0.02, 1.0 / (1.0 + np.exp(2.0 * (z - self.center))))
        
        dh = 0.0
        if len(self.hist) >= 5: dh = h - np.mean(self.hist[-5:-1])
        
        boost = min(0.3, abs(dh) * 2.0) if dh < -self.sens else 0.0
        return min(1.0, base + boost)


class LayerGestalt:
    """Veto Topológico por Capa (Soft-OGP basado en Ledoit-Péché)"""
    def __init__(self, layer, rho=0.8):
        self.layer = layer
        self.rho = rho
        self.U = None
        self.S = None
        self.signal_mask = None
        self.cov_accum = None
        self.mean_accum = None
        self.n_samples = 0
        
        # Reconocer el tipo de capa
        self.is_conv = isinstance(layer, nn.Conv2d)
        if self.is_conv:
            self.k = layer.kernel_size[0]
            self.s = layer.stride[0]
            self.p = layer.padding[0]
            self.d = layer.dilation[0]

    def accumulate(self, x, mask=None):
        """Acumula la covarianza con Welford Batched (corrige sesgo inter-batch).
        
        Welford parallel merge:
          S_AB = S_A + S_B + (n_A * n_B / (n_A + n_B)) * (μ_B - μ_A)(μ_B - μ_A)^T
        """
        with torch.no_grad():
            if self.is_conv:
                # Extraer parches locales: (N, C, H, W) -> (N, C*K*K, L)
                x_unf = F.unfold(x, self.k, self.d, self.p, self.s)
                # Transponer a (N*L, C*K*K) para tratar cada parche como una observación independiente
                x_flat = x_unf.transpose(1, 2).reshape(-1, x_unf.size(1))
            else:
                # Para NLP (3D: [B, Seq, D]) o lineal estandar (2D: [B, D])
                if x.dim() == 3:
                    if mask is not None:
                        x_flat = x[mask.bool()]
                    else:
                        x_flat = x.reshape(-1, x.size(-1))
                else:
                    x_flat = x.view(x.size(0), -1)

            N = x_flat.size(0)
            if N == 0:
                return
            batch_mean = x_flat.mean(dim=0)
            
            # Within-batch scatter (centrado por media del batch)
            x_centered = x_flat - batch_mean
            S_batch = torch.mm(x_centered.t(), x_centered)
            
            if self.mean_accum is None:
                # Primer batch: inicializar acumuladores
                self.mean_accum = batch_mean.clone()
                self.cov_accum = S_batch
                self.n_samples = N
            else:
                # Welford parallel merge
                n_old = self.n_samples
                delta = batch_mean - self.mean_accum
                n_new = n_old + N
                
                # Corrección inter-grupo: captura varianza entre medias de batches
                inter = (n_old * N / n_new) * torch.outer(delta, delta)
                
                # Actualizar media global
                self.mean_accum = self.mean_accum + delta * (N / n_new)
                
                # Actualizar scatter sum: S_AB = S_A + S_B + inter
                self.cov_accum = self.cov_accum + S_batch + inter
                self.n_samples = n_new

    def fit(self, hebbian_decay=0.0):
        """Calcula el SVD de la covarianza acumulada y extrae la Gestalt estructural.
        Si hebbian_decay > 0, aplica decaimiento exponencial en vez de borrar la covarianza,
        permitiendo que la protección acumule información de tareas anteriores."""
        if self.n_samples == 0: return
        cov = self.cov_accum / max(self.n_samples - 1, 1)
        
        orig_dtype = cov.dtype
        cov_f = cov.float()
        cov_f = cov_f + 1e-6 * torch.eye(cov_f.size(0), device=cov_f.device)
        
        U_f, S_f, _ = torch.linalg.svd(cov_f)
        self.U = U_f.to(orig_dtype)
        self.S = S_f.to(orig_dtype)
        
        # Heurística de Ledoit-Péché: La Señal Verdadera supera la media espectral
        self.signal_mask = self.S > self.S.mean()
        
        # Hebbian decay: mantener memoria de tareas pasadas con decaimiento exponencial
        if 0 < hebbian_decay < 1.0:
            self.cov_accum = self.cov_accum * hebbian_decay
            self.n_samples = max(int(self.n_samples * hebbian_decay), 1)
            # mean_accum se preserva para continuidad del Welford
        else:
            self.cov_accum = None
            self.mean_accum = None
            self.n_samples = 0

    def project_gradient(self):
        """Proyecta geométricamente el gradiente de la capa antes de la actualización"""
        if self.U is None or self.layer.weight.grad is None: 
            self.last_blocked_ratio = 0.0
            return
        
        grad = self.layer.weight.grad
        orig_shape = grad.shape
        
        # Aplanar el gradiente. Para Conv2d: (C_out, C_in * K * K). Para Linear: (out_features, in_features)
        grad_flat = grad.view(grad.size(0), -1)
        
        attenuation = torch.ones_like(self.S)
        attenuation[self.signal_mask] = 1.0 - self.rho
        
        # 1. Rotar al espacio de autovectores (Gestalt)
        g_rot = torch.mm(grad_flat, self.U)
        # 2. Atenuar la señal (Orthogonal Soft Projection)
        g_rot = g_rot * attenuation
        # 3. Rotar de vuelta al espacio de parámetros
        g_proj = torch.mm(g_rot, self.U.t())
        
        # Calcular ratio de gradiente bloqueado
        norm_orig = torch.norm(grad_flat)
        norm_proj = torch.norm(g_proj)
        self.last_blocked_ratio = (1.0 - (norm_proj / (norm_orig + 1e-10))).item()
        
        # Restaurar la forma original y sobreescribir el gradiente
        self.layer.weight.grad.copy_(g_proj.view(orig_shape))


class GestaltRegulator:
    """
    Orquestador de ANTENA Gestalt
    =============================
    Combina la Inmunidad Termodinámica (EGPR) y la Inmunidad Topológica (Soft-OGP).
    """
    def __init__(self, model, use_thermo=True, use_topo=True, rho_topo=0.8):
        self.model = model
        self.use_thermo = use_thermo
        self.use_topo = use_topo
        
        self.thermo = EGPRRegulator() if use_thermo else None
        self.layer_gestalts = []
        self.hooks = []
        self.is_calibrating = False
        
        if self.use_topo:
            self._register_layers(rho_topo)

    def _register_layers(self, rho):
        """Engancha el LayerGestalt a todas las capas Conv2D y Linear del modelo"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                lg = LayerGestalt(module, rho=rho)
                self.layer_gestalts.append(lg)
                # Forward hook para interceptar y acumular entradas durante la calibración
                def hook_fn(lg_ref):
                    def fn(mod, inp, out):
                        if self.is_calibrating:
                            lg_ref.accumulate(inp[0])
                    return fn
                self.hooks.append(module.register_forward_hook(hook_fn(lg)))

    def start_calibration(self):
        """Abre la válvula de escucha (acumulación de covarianza topológica)"""
        self.is_calibrating = True

    def end_calibration(self, ents=None):
        """Cierra la válvula y sella la Gestalt (SVD). Calibra EGPR si se requiere."""
        self.is_calibrating = False
        if self.use_topo:
            for lg in self.layer_gestalts: lg.fit()
        if self.use_thermo and ents is not None:
            self.thermo.calibrate(ents)

    def step_plasticity(self, logits, optimizer, base_lr):
        """
        Ejecuta la Defensa Combinada:
        1. Veto Termodinámico: Mide la entropía y modula el LR global.
        2. Veto Topológico: Proyecta los tensores de gradiente para proteger la Gestalt.
        Nota: Llama a este método DESPUÉS de loss.backward() pero ANTES de optimizer.step()
        """
        plast = 1.0
        # 1. Regulación Escalar (Termodinámica)
        if self.use_thermo:
            plast = self.thermo.compute(logits)
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * plast
        
        # 2. Regulación Geométrica (Topológica)
        if self.use_topo:
            for lg in self.layer_gestalts:
                lg.project_gradient()
                
        return plast

    def teardown(self):
        """Limpia los hooks para evitar memory leaks"""
        for h in self.hooks: h.remove()
