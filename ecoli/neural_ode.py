"""
latent_crm_dfba_hybrid.py

Hybrid latent-state CRM / dFBA surrogate
========================================

This script implements a mechanistic-neural hybrid model for learning a
low-dimensional surrogate of dFBA while remaining biologically interpretable.

Core idea
---------
We keep the CRM backbone:

    dX/dt = X * (Yg * ug + Ya * ua - m)
    dG/dt = -X * ug
    dA/dt = X * (alpha * ug - ua)
    dO/dt = -X * uo

but we make the effective parameters state-dependent through a latent state z(t):

    z'(t) = f_theta(X, G, A, O, z)

and learn:
    - Yg(t), Ya(t)
    - maintenance m(t)
    - acetate overflow alpha(t)
    - soft acetate preference gate
    - oxygen effect
    - mild latent modulation of vmax values

This is useful when:
    - fixed yields do not work
    - fixed maintenance does not work
    - hard diauxic switching is too crude
    - dFBA and experiment disagree in late phase

Inputs
------
1. Experimental CSV (required):
       columns: Time, Biomass
2. dFBA CSV (optional but strongly recommended):
       columns like:
           time, biomass, glucose, acetate, oxygen
       column names can be changed in config

Outputs
-------
- trained model
- prediction DataFrame
- plots for biomass and extracellular states
- saved checkpoint

Dependencies
------------
pip install torch pandas numpy matplotlib

Optional
--------
If CUDA is available, training can run on GPU automatically.

Usage
-----
python latent_crm_dfba_hybrid.py

Then edit the paths in main() or pass your own config.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
import json
import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Config
# ============================================================
@dataclass
class DataConfig:
    # Required experiment CSV
    experiment_csv: str = "/Users/edwin/Downloads/yeast_growth_data.csv"
    exp_time_col: str = "Time"
    exp_biomass_col: str = "MeanDensity"

    # Optional dFBA CSV
    dfba_csv: Optional[str] = None
    dfba_time_col: str = "time"
    dfba_biomass_col: str = "biomass"
    dfba_glucose_col: str = "glucose"
    dfba_acetate_col: str = "acetate"
    dfba_oxygen_col: str = "oxygen"

    # Initial states if not fully available from data
    initial_glucose: float = 11.1
    initial_acetate: float = 0.0
    initial_oxygen: float = 50.0

    # Training grid
    time_start: float = 0.0
    time_end: Optional[float] = None
    dt: float = 0.02

    # Scaling / clipping
    biomass_scale: float = 1.0
    substrate_scale: float = 20.0

    # Train/fit
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig:
    latent_dim: int = 4
    hidden_dim: int = 64
    layers: int = 2

    # Initial interpretable values
    init_vmax_glucose: float = 10.0
    init_vmax_acetate: float = 4.0
    init_vmax_oxygen: float = 12.0

    init_km_glucose: float = 0.05
    init_km_acetate: float = 0.05
    init_km_oxygen: float = 0.05

    init_yield_glucose: float = 0.05
    init_yield_acetate: float = 0.03
    init_maintenance: float = 0.01
    init_alpha: float = 0.3

    # Hard biological bounds via softplus / sigmoid maps
    max_yield_glucose: float = 0.30
    max_yield_acetate: float = 0.30
    max_maintenance: float = 0.50
    max_alpha: float = 3.00
    max_vmax_multiplier: float = 2.0

    soft_gate_beta_init: float = 8.0
    gate_threshold_init: float = 0.5

    # Small latent modulation to keep interpretability
    latent_modulation_scale: float = 0.25


@dataclass
class TrainConfig:
    epochs: int = 4000
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 5.0
    print_every: int = 100
    checkpoint_every: int = 500
    checkpoint_dir: str = "./latent_crm_checkpoints"

    # Loss weights
    w_exp_biomass: float = 5.0
    w_dfba_biomass: float = 2.0
    w_dfba_glucose: float = 1.0
    w_dfba_acetate: float = 1.0
    w_dfba_oxygen: float = 0.5

    # Regularization
    w_nonnegative: float = 10.0
    w_smooth_latent: float = 1e-3
    w_smooth_params: float = 1e-3
    w_state_l2: float = 1e-5
    w_latent_l2: float = 1e-5

    # Optional weak priors to keep things sane
    w_prior_params: float = 1e-4

    # Early stopping
    early_stop_patience: int = 800
    min_delta: float = 1e-6


# ============================================================
# Data container
# ============================================================
@dataclass
class TrajectoryData:
    t_grid: torch.Tensor
    exp_t: torch.Tensor
    exp_biomass: torch.Tensor

    dfba_t: Optional[torch.Tensor]
    dfba_biomass: Optional[torch.Tensor]
    dfba_glucose: Optional[torch.Tensor]
    dfba_acetate: Optional[torch.Tensor]
    dfba_oxygen: Optional[torch.Tensor]

    x0: torch.Tensor  # [X0, G0, A0, O0, z0...]

    @property
    def has_dfba(self) -> bool:
        return self.dfba_t is not None


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tensor(x, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def interp1d_torch(
    x_new: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Linear interpolation in torch for 1D x and y.
    x_new shape: [N]
    x shape: [M]
    y shape: [M] or [M, D]
    """
    if y.ndim == 1:
        y = y[:, None]
        squeeze = True
    else:
        squeeze = False

    x_new = torch.clamp(x_new, min=x[0], max=x[-1])

    idx = torch.searchsorted(x, x_new, right=True)
    idx = torch.clamp(idx, 1, len(x) - 1)

    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    w = (x_new - x0) / (x1 - x0 + 1e-12)
    out = y0 + (y1 - y0) * w[:, None]

    if squeeze:
        out = out[:, 0]
    return out


def inverse_softplus(y: float) -> float:
    return math.log(math.exp(y) - 1.0) if y > 1e-8 else -20.0


# ============================================================
# Data loading
# ============================================================
def load_trajectory_data(cfg: DataConfig, latent_dim: int) -> TrajectoryData:
    exp_df = pd.read_csv(cfg.experiment_csv)
    exp_t = exp_df[cfg.exp_time_col].to_numpy(dtype=float)
    exp_x = exp_df[cfg.exp_biomass_col].to_numpy(dtype=float)

    if cfg.time_end is None:
        t_end = float(np.max(exp_t))
    else:
        t_end = float(cfg.time_end)

    t_grid = np.arange(cfg.time_start, t_end + 1e-12, cfg.dt, dtype=float)

    dfba_t = dfba_x = dfba_g = dfba_a = dfba_o = None

    if cfg.dfba_csv is not None and os.path.exists(cfg.dfba_csv):
        ddf = pd.read_csv(cfg.dfba_csv)
        dfba_t = ddf[cfg.dfba_time_col].to_numpy(dtype=float)
        dfba_x = ddf[cfg.dfba_biomass_col].to_numpy(dtype=float)
        dfba_g = ddf[cfg.dfba_glucose_col].to_numpy(dtype=float)
        dfba_a = ddf[cfg.dfba_acetate_col].to_numpy(dtype=float)
        dfba_o = ddf[cfg.dfba_oxygen_col].to_numpy(dtype=float)

        x0 = float(dfba_x[0])
        g0 = float(dfba_g[0])
        a0 = float(dfba_a[0])
        o0 = float(dfba_o[0])
    else:
        x0 = float(exp_x[0])
        g0 = float(cfg.initial_glucose)
        a0 = float(cfg.initial_acetate)
        o0 = float(cfg.initial_oxygen)

    z0 = np.zeros(latent_dim, dtype=float)
    init_state = np.concatenate([[x0, g0, a0, o0], z0])

    return TrajectoryData(
        t_grid=to_tensor(t_grid, cfg.device),
        exp_t=to_tensor(exp_t, cfg.device),
        exp_biomass=to_tensor(exp_x, cfg.device),
        dfba_t=to_tensor(dfba_t, cfg.device) if dfba_t is not None else None,
        dfba_biomass=to_tensor(dfba_x, cfg.device) if dfba_x is not None else None,
        dfba_glucose=to_tensor(dfba_g, cfg.device) if dfba_g is not None else None,
        dfba_acetate=to_tensor(dfba_a, cfg.device) if dfba_a is not None else None,
        dfba_oxygen=to_tensor(dfba_o, cfg.device) if dfba_o is not None else None,
        x0=to_tensor(init_state, cfg.device),
    )


# ============================================================
# Small MLP helper
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2):
        super().__init__()
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(max(1, layers - 1)):
            mods.append(nn.Linear(d, hidden_dim))
            mods.append(nn.Tanh())
            d = hidden_dim
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Hybrid latent CRM model
# ============================================================
class LatentCRMHybrid(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        k = cfg.latent_dim

        # Interpretable base parameters (positive via softplus)
        self.raw_vmax_g = nn.Parameter(torch.tensor(inverse_softplus(cfg.init_vmax_glucose)))
        self.raw_vmax_a = nn.Parameter(torch.tensor(inverse_softplus(cfg.init_vmax_acetate)))
        self.raw_vmax_o = nn.Parameter(torch.tensor(inverse_softplus(cfg.init_vmax_oxygen)))

        self.raw_km_g = nn.Parameter(torch.tensor(inverse_softplus(cfg.init_km_glucose)))
        self.raw_km_a = nn.Parameter(torch.tensor(inverse_softplus(cfg.init_km_acetate)))
        self.raw_km_o = nn.Parameter(torch.tensor(inverse_softplus(cfg.init_km_oxygen)))

        self.raw_yg = nn.Parameter(torch.tensor(0.0))
        self.raw_ya = nn.Parameter(torch.tensor(0.0))
        self.raw_m = nn.Parameter(torch.tensor(0.0))
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.raw_beta = nn.Parameter(torch.tensor(inverse_softplus(cfg.soft_gate_beta_init)))
        self.raw_tau = nn.Parameter(torch.tensor(inverse_softplus(cfg.gate_threshold_init)))

        # Initialize heads close to desired values
        with torch.no_grad():
            self.raw_yg.copy_(torch.tensor(self._inv_sigmoid_scaled(
                cfg.init_yield_glucose, cfg.max_yield_glucose
            )))
            self.raw_ya.copy_(torch.tensor(self._inv_sigmoid_scaled(
                cfg.init_yield_acetate, cfg.max_yield_acetate
            )))
            self.raw_m.copy_(torch.tensor(self._inv_sigmoid_scaled(
                cfg.init_maintenance, cfg.max_maintenance
            )))
            self.raw_alpha.copy_(torch.tensor(self._inv_sigmoid_scaled(
                cfg.init_alpha, cfg.max_alpha
            )))

        # Latent dynamics
        self.latent_rhs = MLP(4 + k, cfg.hidden_dim, k, cfg.layers)

        # Heads for dynamic corrections
        self.param_head = MLP(4 + k, cfg.hidden_dim, 8, cfg.layers)
        # outputs:
        # [dYg, dYa, dM, dAlpha, dVg, dVa, dVo, dTau]

    @staticmethod
    def _inv_sigmoid_scaled(y: float, ymax: float) -> float:
        y = min(max(y / ymax, 1e-6), 1.0 - 1e-6)
        return math.log(y / (1.0 - y))

    def base_params(self) -> Dict[str, torch.Tensor]:
        vmax_g = F.softplus(self.raw_vmax_g)
        vmax_a = F.softplus(self.raw_vmax_a)
        vmax_o = F.softplus(self.raw_vmax_o)

        km_g = F.softplus(self.raw_km_g)
        km_a = F.softplus(self.raw_km_a)
        km_o = F.softplus(self.raw_km_o)

        yg = self.cfg.max_yield_glucose * torch.sigmoid(self.raw_yg)
        ya = self.cfg.max_yield_acetate * torch.sigmoid(self.raw_ya)
        maintenance = self.cfg.max_maintenance * torch.sigmoid(self.raw_m)
        alpha = self.cfg.max_alpha * torch.sigmoid(self.raw_alpha)

        beta = F.softplus(self.raw_beta) + 1e-4
        tau = F.softplus(self.raw_tau)

        return {
            "vmax_g": vmax_g,
            "vmax_a": vmax_a,
            "vmax_o": vmax_o,
            "km_g": km_g,
            "km_a": km_a,
            "km_o": km_o,
            "yg": yg,
            "ya": ya,
            "maintenance": maintenance,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
        }

    def dynamic_params(
            self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        x shape: [..., 4 + latent_dim] = [X, G, A, O, z...]
        """
        base = self.base_params()
        corr = self.param_head(x)

        s = self.cfg.latent_modulation_scale

        # bounded corrections around base
        d_yg = s * torch.tanh(corr[..., 0])
        d_ya = s * torch.tanh(corr[..., 1])
        d_m = s * torch.tanh(corr[..., 2])
        d_alpha = s * torch.tanh(corr[..., 3])
        d_vg = s * torch.tanh(corr[..., 4])
        d_va = s * torch.tanh(corr[..., 5])
        d_vo = s * torch.tanh(corr[..., 6])
        d_tau = s * torch.tanh(corr[..., 7])

        yg = torch.clamp(base["yg"] * (1.0 + d_yg), min=0.0, max=self.cfg.max_yield_glucose)
        ya = torch.clamp(base["ya"] * (1.0 + d_ya), min=0.0, max=self.cfg.max_yield_acetate)
        maintenance = torch.clamp(
            base["maintenance"] * (1.0 + d_m),
            min=0.0,
            max=self.cfg.max_maintenance,
        )
        alpha = torch.clamp(base["alpha"] * (1.0 + d_alpha), min=0.0, max=self.cfg.max_alpha)

        # For tensor-dependent bounds, do it in two steps
        vmax_g = base["vmax_g"] * (1.0 + d_vg)
        vmax_g = torch.clamp(vmax_g, min=1e-6)
        vmax_g = torch.minimum(vmax_g, base["vmax_g"] * self.cfg.max_vmax_multiplier)

        vmax_a = base["vmax_a"] * (1.0 + d_va)
        vmax_a = torch.clamp(vmax_a, min=1e-6)
        vmax_a = torch.minimum(vmax_a, base["vmax_a"] * self.cfg.max_vmax_multiplier)

        vmax_o = base["vmax_o"] * (1.0 + d_vo)
        vmax_o = torch.clamp(vmax_o, min=1e-6)
        vmax_o = torch.minimum(vmax_o, base["vmax_o"] * self.cfg.max_vmax_multiplier)

        tau = base["tau"] * (1.0 + d_tau)
        tau = torch.clamp(tau, min=1e-6, max=100.0)

        out = dict(base)
        out.update({
            "yg": yg,
            "ya": ya,
            "maintenance": maintenance,
            "alpha": alpha,
            "vmax_g": vmax_g,
            "vmax_a": vmax_a,
            "vmax_o": vmax_o,
            "tau": tau,
        })
        return out

    def rhs(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        state shape: [..., 4 + latent_dim]
        returns:
            dstate/dt
            aux params dict
        """
        X = torch.clamp(state[..., 0], min=0.0)
        G = torch.clamp(state[..., 1], min=0.0)
        A = torch.clamp(state[..., 2], min=0.0)
        O = torch.clamp(state[..., 3], min=0.0)
        z = state[..., 4:]

        full = torch.cat([X[..., None], G[..., None], A[..., None], O[..., None], z], dim=-1)
        p = self.dynamic_params(full)
        base = self.base_params()

        ug = p["vmax_g"] * G / (base["km_g"] + G + 1e-12)
        ua_raw = p["vmax_a"] * A / (base["km_a"] + A + 1e-12)
        uo = p["vmax_o"] * O / (base["km_o"] + O + 1e-12)

        # Soft diauxic gate: low acetate usage when glucose is high
        soft_gate = torch.sigmoid(p["beta"] * (p["tau"] - G))
        ua = ua_raw * soft_gate

        # Oxygen effect: smooth saturating multiplier
        oxygen_effect = O / (base["km_o"] + O + 1e-12)

        mu = p["yg"] * ug * oxygen_effect + p["ya"] * ua - p["maintenance"]

        dX = X * mu
        dG = -X * ug
        dA = X * (p["alpha"] * ug - ua)
        dO = -X * uo

        dz = self.latent_rhs(full)

        dstate = torch.cat([dX[..., None], dG[..., None], dA[..., None], dO[..., None], dz], dim=-1)

        aux = {
            "ug": ug,
            "ua": ua,
            "uo": uo,
            "mu": mu,
            "soft_gate": soft_gate,
            "yg": p["yg"],
            "ya": p["ya"],
            "maintenance": p["maintenance"],
            "alpha": p["alpha"],
            "vmax_g": p["vmax_g"],
            "vmax_a": p["vmax_a"],
            "vmax_o": p["vmax_o"],
            "tau": p["tau"],
            "beta": p["beta"],
        }
        return dstate, aux

    def rk4_step(
        self, state: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        k1, a1 = self.rhs(state)
        k2, _ = self.rhs(state + 0.5 * dt * k1)
        k3, _ = self.rhs(state + 0.5 * dt * k2)
        k4, _ = self.rhs(state + dt * k3)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # clamp physical states only
        next_state = torch.cat([
            torch.clamp(next_state[..., 0:1], min=0.0),
            torch.clamp(next_state[..., 1:2], min=0.0),
            torch.clamp(next_state[..., 2:3], min=0.0),
            torch.clamp(next_state[..., 3:4], min=0.0),
            next_state[..., 4:],
        ], dim=-1)

        return next_state, a1

    def simulate(
        self,
        x0: torch.Tensor,
        t_grid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate forward over t_grid.
        """
        assert t_grid.ndim == 1
        state = x0
        states = [state]
        aux_list: List[Dict[str, torch.Tensor]] = []

        for i in range(len(t_grid) - 1):
            dt = float((t_grid[i + 1] - t_grid[i]).item())
            state, aux = self.rk4_step(state, dt)
            states.append(state)
            aux_list.append(aux)

        states = torch.stack(states, dim=0)

        # pad last aux with last valid aux so lengths align
        if aux_list:
            aux_list.append(aux_list[-1])
        else:
            _, aux0 = self.rhs(x0)
            aux_list.append(aux0)

        aux_out = {}
        for key in aux_list[0].keys():
            aux_out[key] = torch.stack([a[key] for a in aux_list], dim=0)

        return {
            "t": t_grid,
            "state": states,
            "biomass": states[:, 0],
            "glucose": states[:, 1],
            "acetate": states[:, 2],
            "oxygen": states[:, 3],
            "latent": states[:, 4:],
            **aux_out,
        }


# ============================================================
# Loss builder
# ============================================================
class HybridLoss:
    def __init__(self, data_cfg: DataConfig, train_cfg: TrainConfig):
        self.data_cfg = data_cfg
        self.cfg = train_cfg

    def __call__(
        self,
        pred: Dict[str, torch.Tensor],
        data: TrajectoryData,
        model: LatentCRMHybrid
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        total = torch.tensor(0.0, device=pred["state"].device)

        # ----------------------------------------------------
        # Experiment biomass fit
        # ----------------------------------------------------
        pred_exp_x = interp1d_torch(data.exp_t, pred["t"], pred["biomass"])
        exp_biomass_loss = torch.mean(
            ((pred_exp_x - data.exp_biomass) / self.data_cfg.biomass_scale) ** 2
        )
        losses["exp_biomass"] = float(exp_biomass_loss.item())
        total = total + self.cfg.w_exp_biomass * exp_biomass_loss

        # ----------------------------------------------------
        # dFBA multi-fidelity fit
        # ----------------------------------------------------
        if data.has_dfba:
            pred_dfba_x = interp1d_torch(data.dfba_t, pred["t"], pred["biomass"])
            pred_dfba_g = interp1d_torch(data.dfba_t, pred["t"], pred["glucose"])
            pred_dfba_a = interp1d_torch(data.dfba_t, pred["t"], pred["acetate"])
            pred_dfba_o = interp1d_torch(data.dfba_t, pred["t"], pred["oxygen"])

            dfba_biomass_loss = torch.mean(
                ((pred_dfba_x - data.dfba_biomass) / self.data_cfg.biomass_scale) ** 2
            )
            dfba_glucose_loss = torch.mean(
                ((pred_dfba_g - data.dfba_glucose) / self.data_cfg.substrate_scale) ** 2
            )
            dfba_acetate_loss = torch.mean(
                ((pred_dfba_a - data.dfba_acetate) / self.data_cfg.substrate_scale) ** 2
            )
            dfba_oxygen_loss = torch.mean(
                ((pred_dfba_o - data.dfba_oxygen) / self.data_cfg.substrate_scale) ** 2
            )

            losses["dfba_biomass"] = float(dfba_biomass_loss.item())
            losses["dfba_glucose"] = float(dfba_glucose_loss.item())
            losses["dfba_acetate"] = float(dfba_acetate_loss.item())
            losses["dfba_oxygen"] = float(dfba_oxygen_loss.item())

            total = total + self.cfg.w_dfba_biomass * dfba_biomass_loss
            total = total + self.cfg.w_dfba_glucose * dfba_glucose_loss
            total = total + self.cfg.w_dfba_acetate * dfba_acetate_loss
            total = total + self.cfg.w_dfba_oxygen * dfba_oxygen_loss
        else:
            losses["dfba_biomass"] = 0.0
            losses["dfba_glucose"] = 0.0
            losses["dfba_acetate"] = 0.0
            losses["dfba_oxygen"] = 0.0

        # ----------------------------------------------------
        # Nonnegativity penalties
        # ----------------------------------------------------
        neg_penalty = (
            F.relu(-pred["biomass"]).pow(2).mean() +
            F.relu(-pred["glucose"]).pow(2).mean() +
            F.relu(-pred["acetate"]).pow(2).mean() +
            F.relu(-pred["oxygen"]).pow(2).mean()
        )
        losses["nonnegative"] = float(neg_penalty.item())
        total = total + self.cfg.w_nonnegative * neg_penalty

        # ----------------------------------------------------
        # Smoothness on latent and dynamic parameters
        # ----------------------------------------------------
        if len(pred["t"]) > 1:
            latent_smooth = (pred["latent"][1:] - pred["latent"][:-1]).pow(2).mean()

            param_smooth = (
                (pred["yg"][1:] - pred["yg"][:-1]).pow(2).mean() +
                (pred["ya"][1:] - pred["ya"][:-1]).pow(2).mean() +
                (pred["maintenance"][1:] - pred["maintenance"][:-1]).pow(2).mean() +
                (pred["alpha"][1:] - pred["alpha"][:-1]).pow(2).mean()
            )
        else:
            latent_smooth = torch.tensor(0.0, device=pred["state"].device)
            param_smooth = torch.tensor(0.0, device=pred["state"].device)

        losses["latent_smooth"] = float(latent_smooth.item())
        losses["param_smooth"] = float(param_smooth.item())

        total = total + self.cfg.w_smooth_latent * latent_smooth
        total = total + self.cfg.w_smooth_params * param_smooth

        # ----------------------------------------------------
        # Mild state / latent regularization
        # ----------------------------------------------------
        state_l2 = pred["state"][:, :4].pow(2).mean()
        latent_l2 = pred["latent"].pow(2).mean()

        losses["state_l2"] = float(state_l2.item())
        losses["latent_l2"] = float(latent_l2.item())

        total = total + self.cfg.w_state_l2 * state_l2
        total = total + self.cfg.w_latent_l2 * latent_l2

        # ----------------------------------------------------
        # Weak priors on base parameters
        # ----------------------------------------------------
        base = model.base_params()
        prior_loss = (
            (base["yg"] - model.cfg.init_yield_glucose) ** 2 +
            (base["ya"] - model.cfg.init_yield_acetate) ** 2 +
            (base["maintenance"] - model.cfg.init_maintenance) ** 2 +
            (base["alpha"] - model.cfg.init_alpha) ** 2
        )
        losses["prior_params"] = float(prior_loss.item())
        total = total + self.cfg.w_prior_params * prior_loss

        losses["total"] = float(total.item())
        return total, losses


# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(
        self,
        model: LatentCRMHybrid,
        data: TrajectoryData,
        data_cfg: DataConfig,
        train_cfg: TrainConfig,
    ):
        self.model = model
        self.data = data
        self.data_cfg = data_cfg
        self.cfg = train_cfg
        self.loss_fn = HybridLoss(data_cfg, train_cfg)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay
        )

        ensure_dir(train_cfg.checkpoint_dir)

    def save_checkpoint(self, epoch: int, best_loss: float) -> str:
        path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "model_config": asdict(self.model.cfg),
            "data_config": asdict(self.data_cfg),
            "train_config": asdict(self.cfg),
        }, path)
        return path

    def fit(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "total": [],
            "exp_biomass": [],
            "dfba_biomass": [],
            "dfba_glucose": [],
            "dfba_acetate": [],
            "dfba_oxygen": [],
        }

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.optim.zero_grad()

            pred = self.model.simulate(self.data.x0, self.data.t_grid)
            total_loss, loss_dict = self.loss_fn(pred, self.data, self.model)

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()

            for k in history.keys():
                history[k].append(loss_dict.get(k, 0.0))

            current = loss_dict["total"]
            improved = current < best_loss - self.cfg.min_delta

            if improved:
                best_loss = current
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % self.cfg.print_every == 0 or epoch == 1:
                print(
                    f"epoch={epoch:5d} "
                    f"total={loss_dict['total']:.6f} "
                    f"expX={loss_dict['exp_biomass']:.6f} "
                    f"dfbaX={loss_dict['dfba_biomass']:.6f} "
                    f"dfbaG={loss_dict['dfba_glucose']:.6f} "
                    f"dfbaA={loss_dict['dfba_acetate']:.6f}"
                )

            if epoch % self.cfg.checkpoint_every == 0:
                self.save_checkpoint(epoch, best_loss)

            if patience_counter >= self.cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_path = self.save_checkpoint(epoch, best_loss)
        print(f"Best model saved to: {final_path}")
        return history


# ============================================================
# Reporting helpers
# ============================================================
def prediction_dataframe(pred: Dict[str, torch.Tensor]) -> pd.DataFrame:
    out = {
        "time": pred["t"].detach().cpu().numpy(),
        "biomass": pred["biomass"].detach().cpu().numpy(),
        "glucose": pred["glucose"].detach().cpu().numpy(),
        "acetate": pred["acetate"].detach().cpu().numpy(),
        "oxygen": pred["oxygen"].detach().cpu().numpy(),
        "mu": pred["mu"].detach().cpu().numpy(),
        "ug": pred["ug"].detach().cpu().numpy(),
        "ua": pred["ua"].detach().cpu().numpy(),
        "uo": pred["uo"].detach().cpu().numpy(),
        "yield_glucose": pred["yg"].detach().cpu().numpy(),
        "yield_acetate": pred["ya"].detach().cpu().numpy(),
        "maintenance": pred["maintenance"].detach().cpu().numpy(),
        "alpha": pred["alpha"].detach().cpu().numpy(),
        "soft_gate": pred["soft_gate"].detach().cpu().numpy(),
    }
    latent = pred["latent"].detach().cpu().numpy()
    for j in range(latent.shape[1]):
        out[f"z{j+1}"] = latent[:, j]
    return pd.DataFrame(out)


def summarize_model(model: LatentCRMHybrid) -> None:
    base = model.base_params()
    print("\n=== Learned base parameters ===")
    for k, v in base.items():
        print(f"{k:14s}: {float(v.detach().cpu().item()):.6f}")


def plot_history(history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(8, 5))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(
    pred: Dict[str, torch.Tensor],
    data: TrajectoryData
) -> None:
    t = pred["t"].detach().cpu().numpy()
    X = pred["biomass"].detach().cpu().numpy()
    G = pred["glucose"].detach().cpu().numpy()
    A = pred["acetate"].detach().cpu().numpy()
    O = pred["oxygen"].detach().cpu().numpy()
    M = pred["maintenance"].detach().cpu().numpy()

    exp_t = data.exp_t.detach().cpu().numpy()
    exp_x = data.exp_biomass.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(t, X, linewidth=2, label="Hybrid model")
    plt.scatter(exp_t, exp_x, s=30, label="Experiment")
    if data.has_dfba:
        plt.plot(
            data.dfba_t.detach().cpu().numpy(),
            data.dfba_biomass.detach().cpu().numpy(),
            linewidth=2,
            label="dFBA",
        )
    plt.xlabel("Time (h)")
    plt.ylabel("Biomass")
    plt.title("Biomass fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(t, G, linewidth=2, label="Glucose")
    plt.plot(t, A, linewidth=2, label="Acetate")
    plt.plot(t, O, linewidth=2, label="Oxygen")
    if data.has_dfba:
        plt.scatter(
            data.dfba_t.detach().cpu().numpy(),
            data.dfba_glucose.detach().cpu().numpy(),
            s=20,
            label="dFBA glucose",
        )
        plt.scatter(
            data.dfba_t.detach().cpu().numpy(),
            data.dfba_acetate.detach().cpu().numpy(),
            s=20,
            label="dFBA acetate",
        )
        plt.scatter(
            data.dfba_t.detach().cpu().numpy(),
            data.dfba_oxygen.detach().cpu().numpy(),
            s=20,
            label="dFBA oxygen",
        )
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("Extracellular states")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(t, pred["yg"].detach().cpu().numpy(), label="Yg")
    plt.plot(t, pred["ya"].detach().cpu().numpy(), label="Ya")
    plt.plot(t, M, label="Maintenance")
    plt.plot(t, pred["alpha"].detach().cpu().numpy(), label="Alpha")
    plt.xlabel("Time (h)")
    plt.ylabel("Effective parameter value")
    plt.title("Learned dynamic parameters")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for j in range(pred["latent"].shape[1]):
        plt.plot(t, pred["latent"][:, j].detach().cpu().numpy(), label=f"z{j+1}")
    plt.xlabel("Time (h)")
    plt.ylabel("Latent state")
    plt.title("Latent dynamics")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# End-to-end runner
# ============================================================
def run_hybrid_training(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Tuple[LatentCRMHybrid, Dict[str, torch.Tensor], pd.DataFrame, Dict[str, List[float]]]:
    set_seed(42)

    data = load_trajectory_data(data_cfg, model_cfg.latent_dim)

    model = LatentCRMHybrid(model_cfg).to(data_cfg.device)
    trainer = Trainer(model, data, data_cfg, train_cfg)

    history = trainer.fit()

    model.eval()
    with torch.no_grad():
        pred = model.simulate(data.x0, data.t_grid)

    pred_df = prediction_dataframe(pred)
    summarize_model(model)

    config_path = os.path.join(train_cfg.checkpoint_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "data_config": asdict(data_cfg),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
            },
            f,
            indent=2,
        )

    pred_path = os.path.join(train_cfg.checkpoint_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")

    plot_history(history)
    plot_predictions(pred, data)

    return model, pred, pred_df, history


# ============================================================
# Example main
# ============================================================
def main():
    # --------------------------------------------------------
    # EDIT THESE PATHS
    # --------------------------------------------------------
    data_cfg = DataConfig(
        experiment_csv="/Users/edwin/Downloads/plot-data (2).csv",
        exp_time_col="Time",
        exp_biomass_col="Biomass",

        # Optional: if you already exported dFBA results to CSV, add it here
        # Must contain columns matching the names below
        dfba_csv=None,
        dfba_time_col="time",
        dfba_biomass_col="biomass",
        dfba_glucose_col="glucose",
        dfba_acetate_col="acetate",
        dfba_oxygen_col="oxygen",

        initial_glucose=11.1,
        initial_acetate=0.0,
        initial_oxygen=50.0,
        dt=0.02,
    )

    model_cfg = ModelConfig(
        latent_dim=4,
        hidden_dim=64,
        layers=3,

        init_vmax_glucose=10.0,
        init_vmax_acetate=10.0,
        init_vmax_oxygen=12.0,

        init_km_glucose=0.05,
        init_km_acetate=0.05,
        init_km_oxygen=0.05,

        init_yield_glucose=0.05,
        init_yield_acetate=0.03,
        init_maintenance=0.01,
        init_alpha=0.3,

        soft_gate_beta_init=8.0,
        gate_threshold_init=0.5,
    )

    train_cfg = TrainConfig(
        epochs=3000,
        lr=1e-3,
        print_every=100,
        checkpoint_every=500,
        checkpoint_dir="latent_crm_output_yeast",

        w_exp_biomass=8.0,
        w_dfba_biomass=2.0,
        w_dfba_glucose=1.0,
        w_dfba_acetate=1.0,
        w_dfba_oxygen=0.5,
    )

    run_hybrid_training(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()