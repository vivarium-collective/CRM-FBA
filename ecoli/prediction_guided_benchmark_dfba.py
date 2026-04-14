"""
prediction_guided_benchmark_dfba.py

Benchmark-anchored prediction-guided dFBA
========================================

This script borrows the stable oxygen treatment from your working benchmark:

- oxygen is treated as an extracellular pool with an availability cap
- oxygen uptake is bounded the benchmark way
- the GSMM chooses the actual oxygen uptake

At the same time:
- glucose is guided by the prediction / hybrid model
- acetate is guided by the prediction / hybrid model, but phase-aware
- growth is NOT tightly tracked
- an optional loose biomass upper cap is used

That makes this much more stable than strict tracking scripts.

Expected predictions.csv columns
--------------------------------
Required:
    time
    biomass
    glucose
    acetate
    oxygen
    yield_glucose
    yield_acetate
    maintenance
    alpha

Optional:
    ug
    ua
    mu
    soft_gate

Experiment CSV columns
----------------------
    Time
    Biomass

Dependencies
------------
pip install cobra numpy pandas matplotlib scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import cobra
from cobra.io import load_model
from cobra.flux_analysis import pfba as cobra_pfba


# ============================================================
# Config
# ============================================================
@dataclass
class PredictionConfig:
    predictions_csv: str = "../latent_crm_output/predictions.csv"
    experiment_csv: Optional[str] = "/Users/edwin/Downloads/plot-data (2).csv"

    exp_time_col: str = "Time"
    exp_biomass_col: str = "Biomass"

    time_col: str = "time"
    biomass_col: str = "biomass"
    glucose_col: str = "glucose"
    acetate_col: str = "acetate"
    oxygen_col: str = "oxygen"

    yg_col: str = "yield_glucose"
    ya_col: str = "yield_acetate"
    maintenance_col: str = "maintenance"
    alpha_col: str = "alpha"

    ug_col: str = "ug"
    ua_col: str = "ua"
    mu_col: str = "mu"
    soft_gate_col: str = "soft_gate"

    use_soft_gate_column: bool = True

    # fallback kinetics if ug/ua absent
    Vg: float = 10.009711
    Va: float = 4.000385
    Kg: float = 0.049527
    Ka: float = 0.050006

    # fallback gate if soft_gate absent
    beta: float = 7.996014
    tau: float = 0.503287

    glucose_uptake_scale: float = 0.8
    acetate_uptake_scale: float = 0.8


@dataclass
class DFBAConfig:
    model_name: str = "textbook"

    ex_glucose: str = "EX_glc__D_e"
    ex_acetate: str = "EX_ac_e"
    ex_oxygen: str = "EX_o2_e"
    biomass_reaction: str = "Biomass_Ecoli_core"

    byproduct_exchanges: Optional[List[str]] = None
    base_medium_exchanges: Optional[List[str]] = None

    use_pfba: bool = False
    step_hours: float = 0.01
    end_time_hours: Optional[float] = None

    initial_biomass: Optional[float] = None
    initial_glucose: Optional[float] = None
    initial_acetate: Optional[float] = None
    initial_oxygen: Optional[float] = None

    # benchmark-style oxygen handling
    vmax_oxygen_benchmark: float = 13.8

    # optional loose biomass cap
    use_loose_biomass_cap: bool = True
    biomass_cap_scale: float = 1.10
    max_biomass_flux: float = 1.0

    # acetate phase gate
    enable_acetate_gate: bool = True
    glucose_gate_threshold: float = 0.5

    # availability caps
    use_availability_caps: bool = True

    # safety
    max_exchange_flux: float = 30.0
    max_mu_update: float = 2.0

    output_dfba_csv: str = "prediction_guided_benchmark_dfba.csv"
    output_crm_csv: str = "prediction_guided_benchmark_crm.csv"
    plot: bool = True
    debug_first_steps: int = 10

    def __post_init__(self):
        if self.byproduct_exchanges is None:
            self.byproduct_exchanges = [
                "EX_ac_e",
                "EX_etoh_e",
                "EX_for_e",
                "EX_lac__D_e",
                "EX_succ_e",
            ]
        if self.base_medium_exchanges is None:
            self.base_medium_exchanges = [
                "EX_h_e",
                "EX_h2o_e",
                "EX_na1_e",
                "EX_k_e",
                "EX_pi_e",
                "EX_so4_e",
                "EX_nh4_e",
                "EX_cl_e",
                "EX_mg2_e",
                "EX_ca2_e",
            ]


# ============================================================
# Utilities
# ============================================================
def normalized_mse(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(((a - b) / float(scale)) ** 2))


def michaelis_menten(S: float, Vmax: float, Km: float) -> float:
    S = max(0.0, float(S))
    return float(Vmax * S / (Km + S + 1e-12))


def logistic_gate(G: float, beta: float, tau: float) -> float:
    return float(1.0 / (1.0 + np.exp(beta * (G - tau))))


def availability_cap_mmol_per_gdw_hr(concentration: float, biomass: float, dt: float) -> float:
    if biomass <= 0.0 or dt <= 0.0:
        return 0.0
    return max(0.0, float(concentration) / (float(biomass) * float(dt)))


def update_concentration_eq7(
    concentration: float,
    exchange_flux: float,
    biomass: float,
    mu: float,
    dt: float,
) -> float:
    substrate_uptake = -float(exchange_flux)
    if abs(mu) < 1e-12:
        return float(concentration) - substrate_uptake * float(biomass) * float(dt)

    return float(concentration) + (
        substrate_uptake * float(biomass) / float(mu)
    ) * (1.0 - np.exp(float(mu) * float(dt)))


def solve_fba(model: cobra.Model, use_pfba: bool):
    return cobra_pfba(model) if use_pfba else model.optimize()


def validate_ids(model: cobra.Model, cfg: DFBAConfig) -> None:
    missing = []
    for rid in [cfg.ex_glucose, cfg.ex_acetate, cfg.ex_oxygen, cfg.biomass_reaction]:
        if rid not in model.reactions:
            missing.append(rid)
    if missing:
        raise ValueError(f"Missing reaction IDs in model: {missing}")


def open_inorganic_base_medium(model: cobra.Model, cfg: DFBAConfig) -> None:
    for ex in model.exchanges:
        ex.lower_bound = 0.0
        ex.upper_bound = 1000.0

    for rid in cfg.base_medium_exchanges:
        if rid in model.reactions:
            rxn = model.reactions.get_by_id(rid)
            rxn.lower_bound = -1000.0
            rxn.upper_bound = 1000.0


# ============================================================
# Prediction-driven controller
# ============================================================
class PredictionController:
    def __init__(self, pred_df: pd.DataFrame, cfg: PredictionConfig):
        self.df = pred_df.sort_values(cfg.time_col).reset_index(drop=True)
        self.cfg = cfg
        self.t = self.df[cfg.time_col].to_numpy(float)

        self.has_ug = cfg.ug_col in self.df.columns
        self.has_ua = cfg.ua_col in self.df.columns
        self.has_mu = cfg.mu_col in self.df.columns
        self.has_soft_gate = cfg.use_soft_gate_column and (cfg.soft_gate_col in self.df.columns)

        self.yg = self.df[cfg.yg_col].to_numpy(float)
        self.ya = self.df[cfg.ya_col].to_numpy(float)
        self.maintenance = self.df[cfg.maintenance_col].to_numpy(float)
        self.alpha = self.df[cfg.alpha_col].to_numpy(float)

        self.ug = self.df[cfg.ug_col].to_numpy(float) if self.has_ug else None
        self.ua = self.df[cfg.ua_col].to_numpy(float) if self.has_ua else None
        self.mu = self.df[cfg.mu_col].to_numpy(float) if self.has_mu else None
        self.soft_gate = self.df[cfg.soft_gate_col].to_numpy(float) if self.has_soft_gate else None

    def interp(self, arr: np.ndarray, t: float) -> float:
        return float(np.interp(t, self.t, arr))

    def gate_at(self, t: float, glucose: float) -> float:
        if self.has_soft_gate:
            return float(np.clip(self.interp(self.soft_gate, t), 0.0, 1.0))
        return logistic_gate(glucose, self.cfg.beta, self.cfg.tau)

    def params_at(self, t: float) -> Dict[str, float]:
        return {
            "Yg": self.interp(self.yg, t),
            "Ya": self.interp(self.ya, t),
            "maintenance": self.interp(self.maintenance, t),
            "alpha": self.interp(self.alpha, t),
        }

    def glucose_target(self, t: float, glucose: float) -> float:
        if self.has_ug:
            ug = max(0.0, self.interp(self.ug, t))
        else:
            ug = michaelis_menten(glucose, self.cfg.Vg, self.cfg.Kg)
        return ug * self.cfg.glucose_uptake_scale

    def acetate_target(self, t: float, glucose: float, acetate: float) -> float:
        gate = self.gate_at(t, glucose)
        if self.has_ua:
            ua = max(0.0, self.interp(self.ua, t))
            if self.has_soft_gate:
                ua *= gate
        else:
            ua = michaelis_menten(acetate, self.cfg.Va, self.cfg.Ka) * gate
        return ua * self.cfg.acetate_uptake_scale

    def mu_cap(self, t: float, glucose: float, acetate: float) -> float:
        p = self.params_at(t)
        ug = self.glucose_target(t, glucose)
        ua = self.acetate_target(t, glucose, acetate)

        if self.has_mu:
            mu = max(0.0, self.interp(self.mu, t))
        else:
            # no oxygen factor here because oxygen is handled benchmark-style in dFBA
            mu = max(0.0, p["Yg"] * ug + p["Ya"] * ua - p["maintenance"])
        return mu

    def row_targets(self, t: float, glucose: float, acetate: float) -> Dict[str, float]:
        p = self.params_at(t)
        return {
            "ug_target": self.glucose_target(t, glucose),
            "ua_target": self.acetate_target(t, glucose, acetate),
            "mu_target": self.mu_cap(t, glucose, acetate),
            "gate": self.gate_at(t, glucose),
            **p,
        }


# ============================================================
# Data loading
# ============================================================
def load_predictions(cfg: PredictionConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.predictions_csv)

    required = [
        cfg.time_col,
        cfg.biomass_col,
        cfg.glucose_col,
        cfg.acetate_col,
        cfg.oxygen_col,
        cfg.yg_col,
        cfg.ya_col,
        cfg.maintenance_col,
        cfg.alpha_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"predictions.csv missing required columns: {missing}")

    return df.sort_values(cfg.time_col).reset_index(drop=True)


def load_experiment(cfg: PredictionConfig) -> Optional[pd.DataFrame]:
    if cfg.experiment_csv is None:
        return None
    exp = pd.read_csv(cfg.experiment_csv)
    if cfg.exp_time_col not in exp.columns or cfg.exp_biomass_col not in exp.columns:
        raise ValueError("Experiment CSV missing required columns.")
    return exp.sort_values(cfg.exp_time_col).reset_index(drop=True)


# ============================================================
# CRM simulation using prediction guidance only
# ============================================================
def simulate_prediction_guided_crm(
    controller: PredictionController,
    d_cfg: DFBAConfig,
    pred_cfg: PredictionConfig,
) -> pd.DataFrame:
    end_time = float(d_cfg.end_time_hours) if d_cfg.end_time_hours is not None else float(controller.t[-1])
    t_eval = np.arange(0.0, end_time + 1e-12, d_cfg.step_hours)

    X0 = float(d_cfg.initial_biomass)
    G0 = float(d_cfg.initial_glucose)
    A0 = float(d_cfg.initial_acetate)
    O0 = float(d_cfg.initial_oxygen)

    def rhs(t, y):
        X, G, A, O = y
        X = max(0.0, X)
        G = max(0.0, G)
        A = max(0.0, A)
        O = max(0.0, O)

        tgt = controller.row_targets(float(t), G, A)

        ug = tgt["ug_target"]
        ua = tgt["ua_target"]

        # oxygen handled benchmark-like in CRM surrogate
        uo = michaelis_menten(O, d_cfg.vmax_oxygen_benchmark, pred_cfg.Ka)  # Ko not separate in config, use dedicated value below
        phi_o = O / (pred_cfg.Ka + O + 1e-12)

        # better use tau/oxygen denominator from learned Ko-like scale if available:
        # since PredictionConfig has no Ko field here, we re-use Kg scale if needed
        # but to preserve a distinct oxygen scale, let's define it explicitly below instead.

        dX = X * (tgt["Yg"] * ug * phi_o + tgt["Ya"] * ua * phi_o - tgt["maintenance"])
        dG = -X * ug
        dA = X * (tgt["alpha"] * ug - ua)
        dO = -X * uo
        return [dX, dG, dA, dO]

    # redefine with explicit oxygen scale from glucose Km fallback if you do not have Ko
    def rhs_with_explicit_oxygen(t, y):
        X, G, A, O = y
        X = max(0.0, X)
        G = max(0.0, G)
        A = max(0.0, A)
        O = max(0.0, O)

        tgt = controller.row_targets(float(t), G, A)

        ug = tgt["ug_target"]
        ua = tgt["ua_target"]
        Ko = pred_cfg.Kg
        uo = michaelis_menten(O, d_cfg.vmax_oxygen_benchmark, Ko)
        phi_o = O / (Ko + O + 1e-12)

        mu = tgt["Yg"] * ug * phi_o + tgt["Ya"] * ua * phi_o - tgt["maintenance"]

        dX = X * mu
        dG = -X * ug
        dA = X * (tgt["alpha"] * ug - ua)
        dO = -X * uo
        return [dX, dG, dA, dO]

    sol = solve_ivp(
        fun=rhs_with_explicit_oxygen,
        t_span=(0.0, end_time),
        y0=[X0, G0, A0, O0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    rows = []
    for i, t in enumerate(sol.t):
        X = max(0.0, sol.y[0, i])
        G = max(0.0, sol.y[1, i])
        A = max(0.0, sol.y[2, i])
        O = max(0.0, sol.y[3, i])

        tgt = controller.row_targets(float(t), G, A)
        Ko = pred_cfg.Kg
        uo = michaelis_menten(O, d_cfg.vmax_oxygen_benchmark, Ko)
        phi_o = O / (Ko + O + 1e-12)
        mu = tgt["Yg"] * tgt["ug_target"] * phi_o + tgt["Ya"] * tgt["ua_target"] * phi_o - tgt["maintenance"]

        rows.append({
            "time": float(t),
            "biomass": X,
            "glucose": G,
            "acetate": A,
            "oxygen": O,
            "mu": mu,
            "ug": tgt["ug_target"],
            "ua": tgt["ua_target"],
            "uo": uo,
            "gate": tgt["gate"],
            "Yg": tgt["Yg"],
            "Ya": tgt["Ya"],
            "maintenance": tgt["maintenance"],
            "alpha": tgt["alpha"],
        })

    return pd.DataFrame(rows)


# ============================================================
# Benchmark-inspired prediction-guided dFBA
# ============================================================
def run_prediction_guided_benchmark_dfba(
    controller: PredictionController,
    d_cfg: DFBAConfig,
) -> pd.DataFrame:
    model = load_model(d_cfg.model_name)
    validate_ids(model, d_cfg)

    end_time = float(d_cfg.end_time_hours) if d_cfg.end_time_hours is not None else float(controller.t[-1])
    time_grid = np.arange(0.0, end_time + 1e-12, d_cfg.step_hours, dtype=float)

    biomass = float(d_cfg.initial_biomass)
    glucose = float(d_cfg.initial_glucose)
    acetate = float(d_cfg.initial_acetate)
    oxygen = float(d_cfg.initial_oxygen)

    rows: List[Dict[str, float]] = []

    for step_i, t in enumerate(time_grid):
        tgt = controller.row_targets(float(t), glucose, acetate)

        ug_pred = min(max(0.0, tgt["ug_target"]), d_cfg.max_exchange_flux)
        ua_pred = min(max(0.0, tgt["ua_target"]), d_cfg.max_exchange_flux)
        mu_pred = min(max(0.0, tgt["mu_target"]), d_cfg.max_biomass_flux)

        # oxygen is benchmark-style only
        uo_benchmark = min(d_cfg.vmax_oxygen_benchmark, d_cfg.max_exchange_flux)

        if d_cfg.use_availability_caps:
            ug_bound = min(ug_pred, availability_cap_mmol_per_gdw_hr(glucose, biomass, d_cfg.step_hours))
            ua_bound = min(ua_pred, availability_cap_mmol_per_gdw_hr(acetate, biomass, d_cfg.step_hours))
            uo_bound = min(uo_benchmark, availability_cap_mmol_per_gdw_hr(oxygen, biomass, d_cfg.step_hours))
        else:
            ug_bound = ug_pred
            ua_bound = ua_pred
            uo_bound = uo_benchmark

        row = {
            "time": float(t),
            "biomass": biomass,
            "glucose": glucose,
            "acetate": acetate,
            "oxygen": oxygen,
            "ug_pred": ug_pred,
            "ua_pred": ua_pred,
            "mu_pred": mu_pred,
            "uo_benchmark": uo_benchmark,
            "ug_bound": ug_bound,
            "ua_bound": ua_bound,
            "uo_bound": uo_bound,
            "gate": tgt["gate"],
            "Yg_used": tgt["Yg"],
            "Ya_used": tgt["Ya"],
            "maintenance_used": tgt["maintenance"],
            "alpha_used": tgt["alpha"],
            "mu_dfba": np.nan,
            "v_glucose": np.nan,
            "v_acetate": np.nan,
            "v_oxygen": np.nan,
            "ug_dfba": np.nan,
            "ua_dfba": np.nan,
            "uo_dfba": np.nan,
            "tracking_mode": "benchmark_guided",
        }

        if t >= end_time:
            rows.append(row)
            break

        with model as m:
            m.objective = d_cfg.biomass_reaction
            open_inorganic_base_medium(m, d_cfg)

            for rid in d_cfg.byproduct_exchanges:
                if rid in m.reactions:
                    r = m.reactions.get_by_id(rid)
                    r.lower_bound = 0.0
                    r.upper_bound = 1000.0

            # glucose: prediction-guided
            m.reactions.get_by_id(d_cfg.ex_glucose).lower_bound = -abs(ug_bound)

            # acetate: phase-aware
            ac_rxn = m.reactions.get_by_id(d_cfg.ex_acetate)
            if d_cfg.enable_acetate_gate and glucose > d_cfg.glucose_gate_threshold:
                ac_rxn.lower_bound = 0.0
                ac_rxn.upper_bound = 1000.0
            else:
                ac_rxn.lower_bound = -abs(ua_bound)
                ac_rxn.upper_bound = 1000.0

            # oxygen: benchmark-style only
            m.reactions.get_by_id(d_cfg.ex_oxygen).lower_bound = -abs(uo_bound)

            # loose biomass upper cap only
            bio_rxn = m.reactions.get_by_id(d_cfg.biomass_reaction)
            bio_rxn.lower_bound = 0.0
            bio_rxn.upper_bound = d_cfg.max_biomass_flux
            if d_cfg.use_loose_biomass_cap:
                loose_cap = min(d_cfg.max_biomass_flux, d_cfg.biomass_cap_scale * mu_pred)
                bio_rxn.upper_bound = max(0.0, loose_cap)

            try:
                sol = solve_fba(m, d_cfg.use_pfba)
            except Exception:
                sol = None

            if sol is None or sol.status != "optimal":
                mu_safe = min(mu_pred, d_cfg.max_mu_update)

                row["mu_dfba"] = mu_safe
                row["v_glucose"] = -ug_bound
                row["v_acetate"] = -ua_bound if not (d_cfg.enable_acetate_gate and glucose > d_cfg.glucose_gate_threshold) else 0.0
                row["v_oxygen"] = -uo_bound
                row["ug_dfba"] = ug_bound
                row["ua_dfba"] = 0.0 if (d_cfg.enable_acetate_gate and glucose > d_cfg.glucose_gate_threshold) else ua_bound
                row["uo_dfba"] = uo_bound
                row["tracking_mode"] = "fallback"

                rows.append(row)

                biomass_new = biomass * np.exp(mu_safe * d_cfg.step_hours)
                glucose = max(0.0, glucose - biomass * ug_bound * d_cfg.step_hours)
                if d_cfg.enable_acetate_gate and glucose > d_cfg.glucose_gate_threshold:
                    acetate = max(0.0, acetate + biomass * (tgt["alpha"] * ug_bound) * d_cfg.step_hours)
                else:
                    acetate = max(0.0, acetate + biomass * (tgt["alpha"] * ug_bound - ua_bound) * d_cfg.step_hours)
                oxygen = max(0.0, oxygen - biomass * uo_bound * d_cfg.step_hours)
                biomass = biomass_new
                continue

            mu = float(sol.objective_value)
            v_glucose = float(sol.fluxes[d_cfg.ex_glucose])
            v_acetate = float(sol.fluxes[d_cfg.ex_acetate])
            v_oxygen = float(sol.fluxes[d_cfg.ex_oxygen])

        mu = max(0.0, min(mu, d_cfg.max_mu_update))

        row["mu_dfba"] = mu
        row["v_glucose"] = v_glucose
        row["v_acetate"] = v_acetate
        row["v_oxygen"] = v_oxygen
        row["ug_dfba"] = max(0.0, -v_glucose)
        row["ua_dfba"] = max(0.0, -v_acetate)
        row["uo_dfba"] = max(0.0, -v_oxygen)
        rows.append(row)

        if step_i < d_cfg.debug_first_steps:
            print(
                f"t={t:.3f}, mu_pred={mu_pred:.4f}, mu_dfba={mu:.4f}, "
                f"ug_pred={ug_pred:.4f}, ug_dfba={row['ug_dfba']:.4f}, "
                f"ua_pred={ua_pred:.4f}, ua_dfba={row['ua_dfba']:.4f}, "
                f"uo_bound={uo_bound:.4f}, uo_dfba={row['uo_dfba']:.4f}"
            )

        biomass_new = biomass * np.exp(mu * d_cfg.step_hours)

        glucose = max(0.0, update_concentration_eq7(glucose, v_glucose, biomass, mu, d_cfg.step_hours))
        acetate = max(0.0, update_concentration_eq7(acetate, v_acetate, biomass, mu, d_cfg.step_hours))
        oxygen = max(0.0, update_concentration_eq7(oxygen, v_oxygen, biomass, mu, d_cfg.step_hours))
        biomass = biomass_new

    return pd.DataFrame(rows)


# ============================================================
# Reporting / plotting
# ============================================================
def compare_to_experiment(
    df: pd.DataFrame,
    experiment_csv: str,
    time_col: str = "Time",
    biomass_col: str = "Biomass",
) -> float:
    exp_df = pd.read_csv(experiment_csv)
    t_exp = exp_df[time_col].to_numpy(float)
    x_exp = exp_df[biomass_col].to_numpy(float)
    x_sim = np.interp(t_exp, df["time"].to_numpy(float), df["biomass"].to_numpy(float))
    nmse = normalized_mse(x_sim, x_exp, scale=1.0)
    print(f"NMSE(sim, experiment) = {nmse:.6f}")
    return nmse


import os

def plot_results(
    pred_df: pd.DataFrame,
    crm_df: pd.DataFrame,
    dfba_df: pd.DataFrame,
    p_cfg: PredictionConfig,
    exp_df: Optional[pd.DataFrame] = None,
) -> None:
    os.makedirs("plots", exist_ok=True)

    t_pred = pred_df[p_cfg.time_col].to_numpy(float)

    plt.figure(figsize=(9, 5))
    plt.plot(t_pred, pred_df[p_cfg.biomass_col], lw=2.5, label="Prediction")
    plt.plot(crm_df["time"], crm_df["biomass"], "--", lw=2.0, label="CRM")
    plt.plot(dfba_df["time"], dfba_df["biomass"], lw=2.0, label="Prediction-guided dFBA")
    if exp_df is not None:
        plt.scatter(
            exp_df[p_cfg.exp_time_col].to_numpy(float),
            exp_df[p_cfg.exp_biomass_col].to_numpy(float),
            s=25,
            label="Experiment",
        )
    plt.xlabel("Time (h)")
    plt.ylabel("Biomass")
    plt.title("Prediction vs CRM vs benchmark-guided dFBA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/biomass.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(t_pred, pred_df[p_cfg.glucose_col], label="Pred glucose")
    plt.plot(t_pred, pred_df[p_cfg.acetate_col], label="Pred acetate")
    plt.plot(t_pred, pred_df[p_cfg.oxygen_col], label="Pred oxygen")
    plt.plot(crm_df["time"], crm_df["glucose"], "--", label="CRM glucose")
    plt.plot(crm_df["time"], crm_df["acetate"], "--", label="CRM acetate")
    plt.plot(crm_df["time"], crm_df["oxygen"], "--", label="CRM oxygen")
    plt.plot(dfba_df["time"], dfba_df["glucose"], ":", label="dFBA glucose")
    plt.plot(dfba_df["time"], dfba_df["acetate"], ":", label="dFBA acetate")
    plt.plot(dfba_df["time"], dfba_df["oxygen"], ":", label="dFBA oxygen")
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("Extracellular states")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/states.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(dfba_df["time"], dfba_df["ug_pred"], label="ug pred")
    plt.plot(dfba_df["time"], dfba_df["ug_dfba"], "--", label="ug dFBA")
    plt.plot(dfba_df["time"], dfba_df["ua_pred"], label="ua pred")
    plt.plot(dfba_df["time"], dfba_df["ua_dfba"], "--", label="ua dFBA")
    plt.plot(dfba_df["time"], dfba_df["uo_bound"], label="uo benchmark bound")
    plt.plot(dfba_df["time"], dfba_df["uo_dfba"], "--", label="uo dFBA")
    plt.xlabel("Time (h)")
    plt.ylabel("Rate")
    plt.title("Prediction-guided uptake vs realized dFBA uptake")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fluxes.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(dfba_df["time"], dfba_df["mu_pred"], label="mu pred / cap source")
    plt.plot(dfba_df["time"], dfba_df["mu_dfba"], "--", label="mu dFBA")
    plt.xlabel("Time (h)")
    plt.ylabel("Growth rate")
    plt.title("Loose biomass cap source vs realized dFBA growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/growth.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    pred_cfg = PredictionConfig(
        predictions_csv="predictions.csv",
        experiment_csv="/Users/edwin/Downloads/plot-data (2).csv",

        time_col="time",
        biomass_col="biomass",
        glucose_col="glucose",
        acetate_col="acetate",
        oxygen_col="oxygen",

        yg_col="yield_glucose",
        ya_col="yield_acetate",
        maintenance_col="maintenance",
        alpha_col="alpha",

        ug_col="ug",
        ua_col="ua",
        mu_col="mu",
        soft_gate_col="soft_gate",

        use_soft_gate_column=True,

        Vg=10.009711,
        Va=4.000385,
        Kg=0.049527,
        Ka=0.050006,
        beta=7.996014,
        tau=0.503287,

        glucose_uptake_scale=1.00,
        acetate_uptake_scale=1.00,
    )

    pred_df = load_predictions(pred_cfg)
    exp_df = load_experiment(pred_cfg)
    controller = PredictionController(pred_df, pred_cfg)

    X0 = float(pred_df[pred_cfg.biomass_col].iloc[0])
    G0 = float(pred_df[pred_cfg.glucose_col].iloc[0])
    A0 = float(pred_df[pred_cfg.acetate_col].iloc[0])
    O0 = float(pred_df[pred_cfg.oxygen_col].iloc[0])
    end_time = float(pred_df[pred_cfg.time_col].iloc[-1])

    d_cfg = DFBAConfig(
        model_name="textbook",
        step_hours=0.01,
        end_time_hours=end_time,
        initial_biomass=X0,
        initial_glucose=G0,
        initial_acetate=A0,
        initial_oxygen=O0,

        use_pfba=False,

        vmax_oxygen_benchmark=13.8,

        use_loose_biomass_cap=True,
        biomass_cap_scale=1,
        max_biomass_flux=1.0,

        enable_acetate_gate=True,
        glucose_gate_threshold=0.5,

        use_availability_caps=True,
        max_exchange_flux=30.0,
        max_mu_update=2.0,

        output_dfba_csv="prediction_guided_benchmark_dfba.csv",
        output_crm_csv="prediction_guided_benchmark_crm.csv",
        plot=True,
        debug_first_steps=10,
    )

    crm_df = simulate_prediction_guided_crm(controller, d_cfg, pred_cfg)
    dfba_df = run_prediction_guided_benchmark_dfba(controller, d_cfg)

    crm_df.to_csv(d_cfg.output_crm_csv, index=False)
    dfba_df.to_csv(d_cfg.output_dfba_csv, index=False)

    print(f"Saved: {d_cfg.output_crm_csv}")
    print(f"Saved: {d_cfg.output_dfba_csv}")

    t_pred = pred_df[pred_cfg.time_col].to_numpy(float)

    pred_x_on_crm = np.interp(crm_df["time"], t_pred, pred_df[pred_cfg.biomass_col].to_numpy(float))
    pred_g_on_crm = np.interp(crm_df["time"], t_pred, pred_df[pred_cfg.glucose_col].to_numpy(float))
    pred_a_on_crm = np.interp(crm_df["time"], t_pred, pred_df[pred_cfg.acetate_col].to_numpy(float))
    pred_o_on_crm = np.interp(crm_df["time"], t_pred, pred_df[pred_cfg.oxygen_col].to_numpy(float))

    pred_x_on_dfba = np.interp(dfba_df["time"], t_pred, pred_df[pred_cfg.biomass_col].to_numpy(float))
    pred_g_on_dfba = np.interp(dfba_df["time"], t_pred, pred_df[pred_cfg.glucose_col].to_numpy(float))
    pred_a_on_dfba = np.interp(dfba_df["time"], t_pred, pred_df[pred_cfg.acetate_col].to_numpy(float))
    pred_o_on_dfba = np.interp(dfba_df["time"], t_pred, pred_df[pred_cfg.oxygen_col].to_numpy(float))

    print("\n=== CRM vs prediction NMSE ===")
    print(f"Biomass NMSE : {normalized_mse(crm_df['biomass'], pred_x_on_crm, scale=1.0):.6f}")
    print(f"Glucose NMSE : {normalized_mse(crm_df['glucose'], pred_g_on_crm, scale=2.0):.6f}")
    print(f"Acetate NMSE : {normalized_mse(crm_df['acetate'], pred_a_on_crm, scale=20.0):.6f}")
    print(f"Oxygen NMSE  : {normalized_mse(crm_df['oxygen'], pred_o_on_crm, scale=20.0):.6f}")

    print("\n=== dFBA vs prediction NMSE ===")
    print(f"Biomass NMSE : {normalized_mse(dfba_df['biomass'], pred_x_on_dfba, scale=1.0):.6f}")
    print(f"Glucose NMSE : {normalized_mse(dfba_df['glucose'], pred_g_on_dfba, scale=20.0):.6f}")
    print(f"Acetate NMSE : {normalized_mse(dfba_df['acetate'], pred_a_on_dfba, scale=20.0):.6f}")
    print(f"Oxygen NMSE  : {normalized_mse(dfba_df['oxygen'], pred_o_on_dfba, scale=20.0):.6f}")

    if exp_df is not None:
        print("\nCRM vs experiment")
        compare_to_experiment(crm_df, pred_cfg.experiment_csv, pred_cfg.exp_time_col, pred_cfg.exp_biomass_col)

        print("dFBA vs experiment")
        compare_to_experiment(dfba_df, pred_cfg.experiment_csv, pred_cfg.exp_time_col, pred_cfg.exp_biomass_col)

    if d_cfg.plot:
        plot_results(pred_df, crm_df, dfba_df, pred_cfg, exp_df=exp_df)


if __name__ == "__main__":
    main()