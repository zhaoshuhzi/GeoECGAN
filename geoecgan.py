#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoECGAN (prototype)
====================

A *research skeleton* that mirrors the figure you provided:

- GAN backbone (Generator + Discriminator)
- Upper branch: NPI-style neural perturbational inference (NPI) built on a surrogate ANN
  to infer whole-brain effective connectivity (EC) from HBN-EEG inputs.
- Lower branch: Geometric Eigenmodes (GEM) built from HCP geometry (or a fallback
  graph Laplacian) to generate geometry-constrained latent patterns.
- Final validation: ChineseEEG text decoding metrics (CER, BLEU).

This code is structured to be *plug-and-play*: you can wire in your actual loaders for
HBN-EEG / HCP / ChineseEEG and swap the stubs with real data.

Author: (c) 2025
License: MIT

-------------------------------------------------------------
What this file provides
-------------------------------------------------------------
1) NPI surrogate model + virtual perturbation to estimate EC via autograd Jacobian.
2) Geometric eigenmodes utilities (Laplace–Beltrami via cotangent L; graph Laplacian fallback).
3) GeoECGAN Generator that fuses NPI-branch latent (z_npi) with GEM-branch latent (z_geo).
4) Discriminator for adversarial training.
5) Losses: GAN loss (hinge), optional KL term (if you enable variational head),
   and cross-branch alignment (cosine) as in the figure.
6) Minimal training loop template.
7) Chinese metrics: CER (%) and BLEU (n-gram with brevity penalty), no external deps.
8) Clear docstrings and TODO markers for real datasets.

This is a *reference scaffold* for rapid prototyping; you will likely adjust
module shapes, hyperparameters, and losses for your project.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def exists(x):
    return x is not None

# -----------------------------
# Metric: Character Error Rate & BLEU (no external packages)
# -----------------------------

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate (Levenshtein distance / |ref|) * 100 (%)."""
    # Classic DP
    r, h = list(ref), list(hyp)
    R, H = len(r), len(h)
    dp = [[0]*(H+1) for _ in range(R+1)]
    for i in range(R+1):
        dp[i][0] = i
    for j in range(H+1):
        dp[0][j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return 100.0 * dp[R][H] / max(1, R)

def _ngram_counts(tokens, n):
    return {tuple(tokens[i:i+n]): 1 + 0*0 for i in range(len(tokens)-n+1)}  # set-like

def _modified_precision(ref_tokens, hyp_tokens, n):
    ref_counts = _ngram_counts(ref_tokens, n)
    hyp_counts = _ngram_counts(hyp_tokens, n)
    if not hyp_counts:
        return 0.0
    match = sum(1 for ng in hyp_counts if ng in ref_counts)
    total = max(1, len(hyp_tokens) - n + 1)
    return match / total

def bleu(ref: str, hyp: str, max_n: int = 4) -> float:
    """Simple BLEU with equal n-gram weights and brevity penalty."""
    ref_tokens = list(ref)  # character-level BLEU for Chinese
    hyp_tokens = list(hyp)
    precisions = []
    for n in range(1, max_n+1):
        precisions.append(_modified_precision(ref_tokens, hyp_tokens, n))
    # geometric mean
    if any(p == 0 for p in precisions):
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    # brevity penalty
    r, c = len(ref_tokens), len(hyp_tokens)
    if c == 0:
        return 0.0
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)
    return bp * geo

# -----------------------------
# Geometric Eigenmodes
# -----------------------------

def _normalize_rows(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = (M**2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return M / norm

class GeometricEigenmodes(nn.Module):
    """
    Build geometric eigenmodes using:
      - Cotangent Laplacian if (V, F) mesh is given
      - Graph Laplacian fallback if adjacency is given

    Returns first `n_modes` eigenvectors (N_vertices x n_modes).
    """
    def __init__(self, n_modes: int = 100):
        super().__init__()
        self.n_modes = n_modes

    @staticmethod
    def cotangent_laplacian(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        V: (N, 3) vertices
        F: (M, 3) faces (long)
        Returns L (N x N) symmetric positive semi-definite.
        """
        N = V.shape[0]
        L = torch.zeros((N, N), dtype=V.dtype, device=V.device)

        def cot(a, b, c):
            # cotangent of angle at 'a' in triangle (a, b, c)
            u = V[b] - V[a]
            v = V[c] - V[a]
            cos = (u @ v) / (u.norm() * v.norm() + 1e-8)
            sin2 = 1 - cos**2
            return cos / (sin2.sqrt() + 1e-8)

        for tri in F.long():
            a, b, c = tri.tolist()
            cab = cot(a, b, c)
            cba = cot(b, a, c)
            ccb = cot(c, a, b)
            # accumulate weights
            L[a, b] -= cab; L[b, a] -= cab
            L[a, c] -= cba; L[c, a] -= cba
            L[b, c] -= ccb; L[c, b] -= ccb

            L[a, a] += cab + cba
            L[b, b] += cab + ccb
            L[c, c] += cba + ccb
        return L

    @staticmethod
    def graph_laplacian(A: torch.Tensor) -> torch.Tensor:
        """A: adjacency (N x N), returns L = D - A."""
        D = torch.diag(A.sum(dim=-1))
        L = D - A
        return L

    def forward(self,
                V: Optional[torch.Tensor] = None,
                F: Optional[torch.Tensor] = None,
                A: Optional[torch.Tensor] = None) -> torch.Tensor:
        if V is not None and F is not None:
            L = self.cotangent_laplacian(V, F)
        elif A is not None:
            L = self.graph_laplacian(A)
        else:
            raise ValueError("Provide (V,F) mesh or adjacency A.")
        # small shift for numerical stability
        L = L + 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
        # eigen decomposition
        evals, evecs = torch.linalg.eigh(L)  # ascending order
        modes = evecs[:, :self.n_modes]
        # normalize modes
        modes = _normalize_rows(modes.T).T
        return modes  # (N_vertices x n_modes)

# -----------------------------
# NPI Surrogate + Virtual Perturbation
# -----------------------------

class NPISurrogate(nn.Module):
    """
    Minimal MLP surrogate for brain dynamics: x[t-3:t] -> x[t+1].
    Used to derive effective connectivity via virtual perturbation (autograd).
    Note: You will likely swap this with your best surrogate (MLP/RNN/VAR).

    Input:  (..., T, R)  T time, R regions
    Predicts next-step (..., R)
    """
    def __init__(self, n_regions: int, hidden: int = 256):
        super().__init__()
        self.n_regions = n_regions
        self.net = nn.Sequential(
            nn.Linear(3 * n_regions, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, n_regions),
        )

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        # x_window: (..., 3, R)
        B = x_window.shape[:-2]
        z = x_window.reshape(*B, 3 * self.n_regions)
        y = self.net(z)
        return y  # (..., R)

def npi_virtual_ec(surrogate: NPISurrogate,
                   x_window: torch.Tensor,
                   eps: float = 1e-2) -> torch.Tensor:
    """
    Compute one-to-all EC by perturbing each region (Jacobian via autograd).
    Returns EC matrix: (R x R) where EC[i, j] ~ dy_j / dx_i
    """
    surrogate.eval()
    R = surrogate.n_regions
    x_window = x_window.requires_grad_(True)
    y = surrogate(x_window)  # (..., R)
    # Sum over batch/time dims to get scalar per output dim
    EC = []
    for j in range(R):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[..., j] = 1.0
        grads = torch.autograd.grad(
            outputs=y, inputs=x_window, grad_outputs=grad_outputs,
            retain_graph=True, create_graph=False, allow_unused=True
        )[0]  # (..., 3, R)
        if grads is None:
            grads = torch.zeros_like(x_window)
        # Take last time slice's gradient for simplicity
        jac = grads[..., -1, :]  # (..., R)
        EC.append(jac.mean(dim=tuple(range(len(jac.shape)-1))))  # (R,)
    EC = torch.stack(EC, dim=0)  # (R, R): row=j (target), col=i (source)
    return EC

# -----------------------------
# GeoECGAN: Generator & Discriminator
# -----------------------------

class GeneratorGeoECGAN(nn.Module):
    """
    Fuse two branches:
      - NPI branch: from EEG windows -> EC -> latent z_npi
      - GEM branch: from geometry modes + seed -> latent z_geo
    Output: latent Z and an optional reconstruction (for auxiliary losses).
    """
    def __init__(self, n_regions: int, n_modes: int, latent_dim: int = 128):
        super().__init__()
        self.n_regions = n_regions
        self.n_modes = n_modes
        self.latent_dim = latent_dim

        # NPI branch encoders
        self.npi_surrogate = NPISurrogate(n_regions=n_regions, hidden=256)
        self.npi_proj = nn.Sequential(
            nn.Linear(n_regions * n_regions, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

        # GEM branch encoders (modes are treated as fixed basis externally)
        self.geo_proj = nn.Sequential(
            nn.Linear(n_modes, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )

        # Optional decoder to reconstruct EEG (aux loss)
        self.recon = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, n_regions)
        )

    def forward(self, x_window: torch.Tensor, modes: torch.Tensor,
                mode_coeff: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        x_window: (B, 3, R)
        modes:   (R, n_modes) geometric eigenmodes
        mode_coeff: (B, n_modes) optional coefficients; if None, sample N(0,1)

        Returns:
          {
            'z_npi', 'z_geo', 'z', 'ec', 'recon'
          }
        """
        B, _, R = x_window.shape
        # ---- NPI branch
        with torch.set_grad_enabled(self.training):
            ec = npi_virtual_ec(self.npi_surrogate, x_window)  # (R, R)
        z_npi = self.npi_proj(ec.reshape(R*R).unsqueeze(0)).repeat(B, 1)

        # ---- GEO branch
        if mode_coeff is None:
            mode_coeff = torch.randn(B, modes.shape[1], device=modes.device, dtype=modes.dtype)
        # project coefficients through eigenmodes to regional pattern, then compress
        geo_pattern = mode_coeff @ modes.T  # (B, R)
        z_geo = self.geo_proj(mode_coeff)   # (B, latent_dim)

        # ---- Fuse (simple add; could concat + MLP)
        z = F.normalize(z_npi + z_geo, dim=-1)

        # ---- Optional reconstruction for aux task
        recon = self.recon(z)

        return {'z_npi': z_npi, 'z_geo': z_geo, 'z': z, 'ec': ec, 'recon': recon}

class Discriminator(nn.Module):
    """Simple MLP discriminator on latent Z."""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

# -----------------------------
# Losses
# -----------------------------

def hinge_d_loss(D_real: torch.Tensor, D_fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1. - D_real).mean() + F.relu(1. + D_fake).mean()

def hinge_g_loss(D_fake: torch.Tensor) -> torch.Tensor:
    return -D_fake.mean()

def cosine_align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()

# -----------------------------
# Data Stubs (replace with real loaders)
# -----------------------------

class HBN_EEG_WindowDataset(torch.utils.data.Dataset):
    """
    Stub dataset. Replace `__getitem__` to load a (3, R) EEG window from HBN-EEG.
    """
    def __init__(self, n_samples=64, n_regions=360):
        self.n = n_samples
        self.R = n_regions

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(3, self.R) * 0.1  # (3, R)
        return x

# -----------------------------
# Training Step (single-iter demo)
# -----------------------------

@dataclass
class TrainConfig:
    n_regions: int = 360         # MMP atlas in HCP (example)
    n_modes: int = 100
    latent_dim: int = 128
    batch_size: int = 8
    lr: float = 2e-4
    device: str = "cpu"

def single_train_step(cfg: TrainConfig,
                      generator: GeneratorGeoECGAN,
                      discriminator: Discriminator,
                      modes: torch.Tensor,
                      batch: torch.Tensor,
                      opt_g, opt_d) -> Dict[str, float]:
    generator.train(); discriminator.train()
    batch = batch.to(cfg.device)  # (B, 3, R)

    # ---- Generator forward
    out = generator(batch, modes.to(cfg.device))
    z = out['z']
    # Sample "real" latent codes (you may replace with text/encoder outputs)
    z_real = torch.randn_like(z)

    # ---- Discriminator step
    opt_d.zero_grad(set_to_none=True)
    d_real = discriminator(z_real.detach())
    d_fake = discriminator(z.detach())
    loss_d = hinge_d_loss(d_real, d_fake)
    loss_d.backward()
    opt_d.step()

    # ---- Generator step
    opt_g.zero_grad(set_to_none=True)
    d_fake = discriminator(z)
    # GAN + alignment (z_npi, z_geo) + optional recon loss
    loss_g = hinge_g_loss(d_fake) \
             + 0.1 * cosine_align(out['z_npi'], out['z_geo']) \
             + 0.1 * F.mse_loss(out['recon'], batch[:, -1, :])  # reconstruct last frame
    loss_g.backward()
    opt_g.step()

    return {
        "loss_d": float(loss_d.item()),
        "loss_g": float(loss_g.item())
    }

# -----------------------------
# ChineseEEG evaluation utilities
# -----------------------------

def evaluate_chinese(text_refs, text_hyps) -> Dict[str, float]:
    """
    Compute CER (%) and BLEU for lists of references/hypotheses.
    Character-level for both metrics (suitable for Chinese).
    """
    assert len(text_refs) == len(text_hyps)
    cer_vals, bleu_vals = [], []
    for r, h in zip(text_refs, text_hyps):
        cer_vals.append(cer(r, h))
        bleu_vals.append(bleu(r, h, max_n=4))
    return {
        "CER(%)": sum(cer_vals)/max(1, len(cer_vals)),
        "BLEU": sum(bleu_vals)/max(1, len(bleu_vals))
    }

# -----------------------------
# Demo main
# -----------------------------

def demo():
    set_seed(7)
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    # 1) Build geometric modes (here: graph Laplacian fallback)
    #    Replace A with your HCP-derived surface adjacency or use (V,F) mesh.
    R = cfg.n_regions
    A = torch.rand(R, R)
    A = (A + A.T) / 2
    A.fill_diagonal_(0.0)
    gem = GeometricEigenmodes(n_modes=cfg.n_modes)
    modes = gem(A=A).to(device)  # (R, n_modes)

    # 2) Models
    G = GeneratorGeoECGAN(n_regions=R, n_modes=cfg.n_modes, latent_dim=cfg.latent_dim).to(device)
    D = Discriminator(latent_dim=cfg.latent_dim).to(device)

    # 3) Optims
    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    # 4) Data
    ds = HBN_EEG_WindowDataset(n_samples=16, n_regions=R)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # 5) One epoch (few steps) demo
    logs = []
    for batch in dl:
        log = single_train_step(cfg, G, D, modes, batch, opt_g, opt_d)
        logs.append(log)
    print("Training logs (first 3):", logs[:3])

    # 6) ChineseEEG evaluation demo (placeholder strings)
    refs = ["我爱大脑研究", "语言网络与几何模态"]
    hyps = ["我爱脑研究", "语言网络及几何模态"]
    metrics = evaluate_chinese(refs, hyps)
    print("ChineseEEG metrics (demo):", metrics)

if __name__ == "__main__":
    demo()
