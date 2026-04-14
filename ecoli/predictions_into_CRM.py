"""
crm_from_predictions_simulator.py

Build and simulate an interpretable CRM from a hybrid-model predictions.csv.

What this script does
---------------------
1. Reads predictions.csv from your latent/hybrid model
2. Extracts effective CRM parameters from the prediction trajectories
3. Builds either:
      A) constant-parameter CRM
      B) phase-based CRM (glucose phase / acetate phase)
4. Simulates the CRM forward in time
5. Compares CRM vs reference predictions
6. Optionally overlays experimental biomass CSV

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
    soft_gate

Expected experiment CSV columns
-------------------------------
    Time
    Biomass

Usage
-----
Edit the paths in main() and run:

    python crm_from_predictions_simulator.py

Notes
-----
- This script does NOT retrain anything.
- It converts learned hybrid outputs into a simpler CRM and simulates that CRM.
- This is the right next step if you want to move from the hybrid model back to
  an interpretable reduced system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# Data containers
# ============================================================
@dataclass
class CRMPhaseParameters:
    Vg: float
    Va: float
    Vo: float
    Kg: float
    Ka: float
    Ko: float
    Yg: float
    Ya: float
    maintenance: float
    alpha: float
    tau: float
    beta: float


@dataclass
class CRMSimulationResults:
    time: np.ndarray
    biomass: np.ndarray
    glucose: np.ndarray
    acetate: np.ndarray
    oxygen: np.ndarray
    mu: np.ndarray
    ug: np.ndarray
    ua: np.ndarray
    uo: np.ndarray
    phase: np.ndarray


# ============================================================
# Config
# ============================================================
@dataclass
class CRMConfig:
    # Input files
    predictions_csv: str = "predictions.csv"
    experiment_csv: Optional[str] = "/Users/edwin/Downloads/plot-data (2).csv"

    # Experiment column names
    exp_time_col: str = "Time"
    exp_biomass_col: str = "Biomass"

    # Prediction column names
    time_col: str = "time"
    biomass_col: str = "biomass"
    glucose_col: str = "glucose"
    acetate_col: str = "acetate"
    oxygen_col: str = "oxygen"
    yg_col: str = "yield_glucose"
    ya_col: str = "yield_acetate"
    maintenance_col: str = "maintenance"
    alpha_col: str = "alpha"
    soft_gate_col: str = "soft_gate"

    # Simulation control
    use_phase_based_crm: bool = True
    step_hours: float = 0.01
    end_time_hours: Optional[float] = None

    # Phase split
    glucose_phase_threshold: float = 0.5

    # If soft_gate column exists, use it to estimate tau/beta
    estimate_gate_from_soft_gate: bool = False

    # Manual fallback kinetics if not inferable from predictions
    default_Kg: float = 0.05
    default_Ka: float = 0.05
    default_Ko: float = 0.05

    # If predictions.csv has ug/ua/uo columns, use them
    ug_col: str = "ug"
    ua_col: str = "ua"
    uo_col: str = "uo"

    # Plotting
    plot: bool = True


# ============================================================
# Helpers
# ============================================================
def safe_mean(x: np.ndarray, fallback: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(fallback)
    return float(np.mean(x))


def safe_median(x: np.ndarray, fallback: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(fallback)
    return float(np.median(x))


def infer_vmax_from_uptake_and_substrate(
    uptake: np.ndarray,
    substrate: np.ndarray,
    K: float,
    quantile: float = 0.90,
) -> float:
    """
    Infer Vmax from:
        u = V * S/(K + S)
    so:
        V = u * (K + S)/S
    Uses a robust quantile instead of a raw max.
    """
    substrate = np.asarray(substrate, dtype=float)
    uptake = np.asarray(uptake, dtype=float)

    mask = np.isfinite(substrate) & np.isfinite(uptake) & (substrate > 1e-8) & (uptake >= 0.0)
    if not np.any(mask):
        return 1.0

    v_candidates = uptake[mask] * (K + substrate[mask]) / substrate[mask]
    v_candidates = v_candidates[np.isfinite(v_candidates) & (v_candidates >= 0.0)]
    if v_candidates.size == 0:
        return 1.0

    return float(np.quantile(v_candidates, quantile))


def logistic_gate(G: np.ndarray, tau: float, beta: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(beta * (G - tau)))


def normalized_mse(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(((a - b) / float(scale)) ** 2))


# ============================================================
# Load data
# ============================================================
def load_predictions(cfg: CRMConfig) -> pd.DataFrame:
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

    df = df.sort_values(cfg.time_col).reset_index(drop=True)
    return df


def load_experiment(cfg: CRMConfig) -> Optional[pd.DataFrame]:
    if cfg.experiment_csv is None:
        return None
    exp = pd.read_csv(cfg.experiment_csv)
    if cfg.exp_time_col not in exp.columns or cfg.exp_biomass_col not in exp.columns:
        raise ValueError("Experiment CSV missing required columns.")
    exp = exp.sort_values(cfg.exp_time_col).reset_index(drop=True)
    return exp


# ============================================================
# Parameter extraction
# ============================================================
def extract_constant_parameters(df: pd.DataFrame, cfg: CRMConfig) -> CRMPhaseParameters:
    G = df[cfg.glucose_col].to_numpy(float)
    A = df[cfg.acetate_col].to_numpy(float)
    O = df[cfg.oxygen_col].to_numpy(float)

    Yg = safe_median(df[cfg.yg_col].to_numpy(float), fallback=0.05)
    Ya = safe_median(df[cfg.ya_col].to_numpy(float), fallback=0.03)
    maintenance = safe_median(df[cfg.maintenance_col].to_numpy(float), fallback=0.01)
    alpha = safe_median(df[cfg.alpha_col].to_numpy(float), fallback=0.3)

    Kg = cfg.default_Kg
    Ka = cfg.default_Ka
    Ko = cfg.default_Ko

    if cfg.ug_col in df.columns:
        ug = np.maximum(0.0, df[cfg.ug_col].to_numpy(float))
        Vg = infer_vmax_from_uptake_and_substrate(ug, G, Kg)
    else:
        Vg = 10.0

    if cfg.ua_col in df.columns:
        ua = np.maximum(0.0, df[cfg.ua_col].to_numpy(float))
        Vg_dummy_gate = np.ones_like(ua)
        Va = infer_vmax_from_uptake_and_substrate(ua / np.maximum(Vg_dummy_gate, 1e-8), A, Ka)
    else:
        Va = 4.0

    if cfg.uo_col in df.columns:
        uo = np.maximum(0.0, df[cfg.uo_col].to_numpy(float))
        Vo = infer_vmax_from_uptake_and_substrate(uo, O, Ko)
    else:
        Vo = 12.0

    tau = cfg.glucose_phase_threshold
    beta = 8.0

    return CRMPhaseParameters(
        Vg=Vg, Va=Va, Vo=Vo,
        Kg=Kg, Ka=Ka, Ko=Ko,
        Yg=Yg, Ya=Ya,
        maintenance=maintenance,
        alpha=alpha,
        tau=tau, beta=beta,
    )


def extract_phase_parameters(df: pd.DataFrame, cfg: CRMConfig) -> Tuple[CRMPhaseParameters, CRMPhaseParameters]:
    glucose = df[cfg.glucose_col].to_numpy(float)
    acetate = df[cfg.acetate_col].to_numpy(float)
    oxygen = df[cfg.oxygen_col].to_numpy(float)

    phase1_mask = glucose > cfg.glucose_phase_threshold
    phase2_mask = ~phase1_mask

    if not np.any(phase1_mask):
        phase1_mask[:] = True
    if not np.any(phase2_mask):
        phase2_mask[:] = True

    def build(mask: np.ndarray, tau_value: float) -> CRMPhaseParameters:
        sub = df.loc[mask].copy()

        G = sub[cfg.glucose_col].to_numpy(float)
        A = sub[cfg.acetate_col].to_numpy(float)
        O = sub[cfg.oxygen_col].to_numpy(float)

        Yg = safe_median(sub[cfg.yg_col].to_numpy(float), fallback=0.05)
        Ya = safe_median(sub[cfg.ya_col].to_numpy(float), fallback=0.03)
        maintenance = safe_median(sub[cfg.maintenance_col].to_numpy(float), fallback=0.01)
        alpha = safe_median(sub[cfg.alpha_col].to_numpy(float), fallback=0.3)

        Kg = cfg.default_Kg
        Ka = cfg.default_Ka
        Ko = cfg.default_Ko

        if cfg.ug_col in sub.columns:
            ug = np.maximum(0.0, sub[cfg.ug_col].to_numpy(float))
            Vg = infer_vmax_from_uptake_and_substrate(ug, G, Kg)
        else:
            Vg = 10.0

        if cfg.ua_col in sub.columns:
            ua = np.maximum(0.0, sub[cfg.ua_col].to_numpy(float))
            Va = infer_vmax_from_uptake_and_substrate(ua, A, Ka)
        else:
            Va = 4.0

        if cfg.uo_col in sub.columns:
            uo = np.maximum(0.0, sub[cfg.uo_col].to_numpy(float))
            Vo = infer_vmax_from_uptake_and_substrate(uo, O, Ko)
        else:
            Vo = 12.0

        beta = 8.0
        if cfg.estimate_gate_from_soft_gate and cfg.soft_gate_col in sub.columns:
            beta = 8.0

        return CRMPhaseParameters(
            Vg=Vg, Va=Va, Vo=Vo,
            Kg=Kg, Ka=Ka, Ko=Ko,
            Yg=Yg, Ya=Ya,
            maintenance=maintenance,
            alpha=alpha,
            tau=tau_value, beta=beta,
        )

    p1 = build(phase1_mask, cfg.glucose_phase_threshold)
    p2 = build(phase2_mask, cfg.glucose_phase_threshold)
    return p1, p2


# ============================================================
# CRM simulation
# ============================================================
def simulate_crm_constant(
    params: CRMPhaseParameters,
    X0: float,
    G0: float,
    A0: float,
    O0: float,
    end_time_hours: float,
    step_hours: float,
) -> CRMSimulationResults:
    t_eval = np.arange(0.0, end_time_hours + 1e-12, step_hours, dtype=float)

    def rhs(_, y):
        X, G, A, O = y
        X = max(0.0, X)
        G = max(0.0, G)
        A = max(0.0, A)
        O = max(0.0, O)

        ug = params.Vg * G / (params.Kg + G + 1e-12)
        ua_raw = params.Va * A / (params.Ka + A + 1e-12)
        gate = 1.0 / (1.0 + np.exp(params.beta * (G - params.tau)))
        ua = ua_raw * gate
        uo = params.Vo * O / (params.Ko + O + 1e-12)

        oxygen_effect = O / (params.Ko + O + 1e-12)
        mu = params.Yg * ug * oxygen_effect + params.Ya * ua - params.maintenance

        dX = X * mu
        dG = -X * ug
        dA = X * (params.alpha * ug - ua)
        dO = -X * uo
        return [dX, dG, dA, dO]

    sol = solve_ivp(
        rhs,
        (0.0, float(end_time_hours)),
        [float(X0), float(G0), float(A0), float(O0)],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    X = np.maximum(0.0, sol.y[0])
    G = np.maximum(0.0, sol.y[1])
    A = np.maximum(0.0, sol.y[2])
    O = np.maximum(0.0, sol.y[3])

    ug = params.Vg * G / (params.Kg + G + 1e-12)
    ua_raw = params.Va * A / (params.Ka + A + 1e-12)
    gate = 1.0 / (1.0 + np.exp(params.beta * (G - params.tau)))
    ua = ua_raw * gate
    uo = params.Vo * O / (params.Ko + O + 1e-12)
    oxygen_effect = O / (params.Ko + O + 1e-12)
    mu = params.Yg * ug * oxygen_effect + params.Ya * ua - params.maintenance
    phase = np.zeros_like(X, dtype=int)

    return CRMSimulationResults(
        time=t_eval, biomass=X, glucose=G, acetate=A, oxygen=O,
        mu=mu, ug=ug, ua=ua, uo=uo, phase=phase,
    )


def simulate_crm_phase_based(
    p1: CRMPhaseParameters,
    p2: CRMPhaseParameters,
    X0: float,
    G0: float,
    A0: float,
    O0: float,
    end_time_hours: float,
    step_hours: float,
    threshold: float,
) -> CRMSimulationResults:
    t_grid = np.arange(0.0, end_time_hours + 1e-12, step_hours, dtype=float)

    X = np.zeros_like(t_grid)
    G = np.zeros_like(t_grid)
    A = np.zeros_like(t_grid)
    O = np.zeros_like(t_grid)
    mu = np.zeros_like(t_grid)
    ug = np.zeros_like(t_grid)
    ua = np.zeros_like(t_grid)
    uo = np.zeros_like(t_grid)
    phase = np.zeros_like(t_grid, dtype=int)

    X[0], G[0], A[0], O[0] = float(X0), float(G0), float(A0), float(O0)

    def step_rhs(y, params):
        Xv, Gv, Av, Ov = y
        Xv = max(0.0, Xv)
        Gv = max(0.0, Gv)
        Av = max(0.0, Av)
        Ov = max(0.0, Ov)

        ugv = params.Vg * Gv / (params.Kg + Gv + 1e-12)
        ua_rawv = params.Va * Av / (params.Ka + Av + 1e-12)
        gatev = 1.0 / (1.0 + np.exp(params.beta * (Gv - params.tau)))
        uav = ua_rawv * gatev
        uov = params.Vo * Ov / (params.Ko + Ov + 1e-12)

        oxygen_effectv = Ov / (params.Ko + Ov + 1e-12)
        muv = params.Yg * ugv * oxygen_effectv + params.Ya * uav - params.maintenance

        dX = Xv * muv
        dG = -Xv * ugv
        dA = Xv * (params.alpha * ugv - uav)
        dO = -Xv * uov
        return np.array([dX, dG, dA, dO], dtype=float), (muv, ugv, uav, uov)

    for i in range(len(t_grid) - 1):
        y = np.array([X[i], G[i], A[i], O[i]], dtype=float)
        params = p1 if G[i] > threshold else p2
        phase[i] = 1 if G[i] > threshold else 2

        k1, aux1 = step_rhs(y, params)
        k2, _ = step_rhs(y + 0.5 * step_hours * k1, params)
        k3, _ = step_rhs(y + 0.5 * step_hours * k2, params)
        k4, _ = step_rhs(y + step_hours * k3, params)

        y_next = y + (step_hours / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_next = np.maximum(y_next, 0.0)

        X[i + 1], G[i + 1], A[i + 1], O[i + 1] = y_next
        mu[i], ug[i], ua[i], uo[i] = aux1

    last_params = p1 if G[-1] > threshold else p2
    phase[-1] = 1 if G[-1] > threshold else 2
    _, aux_last = step_rhs(np.array([X[-1], G[-1], A[-1], O[-1]]), last_params)
    mu[-1], ug[-1], ua[-1], uo[-1] = aux_last

    return CRMSimulationResults(
        time=t_grid, biomass=X, glucose=G, acetate=A, oxygen=O,
        mu=mu, ug=ug, ua=ua, uo=uo, phase=phase,
    )


# ============================================================
# Reporting
# ============================================================
def print_params(name: str, p: CRMPhaseParameters) -> None:
    print(f"\n=== {name} ===")
    print(f"Vg           : {p.Vg:.6f}")
    print(f"Va           : {p.Va:.6f}")
    print(f"Vo           : {p.Vo:.6f}")
    print(f"Kg           : {p.Kg:.6f}")
    print(f"Ka           : {p.Ka:.6f}")
    print(f"Ko           : {p.Ko:.6f}")
    print(f"Yg           : {p.Yg:.6f}")
    print(f"Ya           : {p.Ya:.6f}")
    print(f"maintenance  : {p.maintenance:.6f}")
    print(f"alpha        : {p.alpha:.6f}")
    print(f"tau          : {p.tau:.6f}")
    print(f"beta         : {p.beta:.6f}")


def plot_results(
    ref_df: pd.DataFrame,
    crm: CRMSimulationResults,
    cfg: CRMConfig,
    exp_df: Optional[pd.DataFrame] = None,
    prefix: str = "results",
) -> None:
    t_ref = ref_df[cfg.time_col].to_numpy(float)
    x_ref = ref_df[cfg.biomass_col].to_numpy(float)
    g_ref = ref_df[cfg.glucose_col].to_numpy(float)
    a_ref = ref_df[cfg.acetate_col].to_numpy(float)
    o_ref = ref_df[cfg.oxygen_col].to_numpy(float)

    # ==============================
    # 1. BIOMASS
    # ==============================
    plt.figure(figsize=(8, 5))
    plt.plot(t_ref, x_ref, lw=2.5, label="Hybrid/reference biomass")
    plt.plot(crm.time, crm.biomass, lw=2.0, label="CRM biomass")

    if exp_df is not None:
        plt.scatter(
            exp_df[cfg.exp_time_col].to_numpy(float),
            exp_df[cfg.exp_biomass_col].to_numpy(float),
            s=25,
            label="Experiment biomass"
        )

    plt.xlabel("Time (h)")
    plt.ylabel("Biomass")
    plt.title("Biomass: CRM vs reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_biomass.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================
    # 2. EXTRACELLULAR STATES
    # ==============================
    plt.figure(figsize=(8, 5))
    plt.plot(t_ref, g_ref, lw=2.0, label="Reference glucose")
    plt.plot(t_ref, a_ref, lw=2.0, label="Reference acetate")
    plt.plot(t_ref, o_ref, lw=2.0, label="Reference oxygen")

    plt.plot(crm.time, crm.glucose, "--", lw=2.0, label="CRM glucose")
    plt.plot(crm.time, crm.acetate, "--", lw=2.0, label="CRM acetate")
    plt.plot(crm.time, crm.oxygen, "--", lw=2.0, label="CRM oxygen")

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("Extracellular dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_states.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================
    # 3. CRM RATES
    # ==============================
    plt.figure(figsize=(8, 4))
    plt.plot(crm.time, crm.mu, label="mu")
    plt.plot(crm.time, crm.ug, label="ug")
    plt.plot(crm.time, crm.ua, label="ua")
    plt.plot(crm.time, crm.uo, label="uo")

    plt.xlabel("Time (h)")
    plt.ylabel("Rate")
    plt.title("CRM rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_rates.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main bridge: predictions -> CRM -> simulate
# ============================================================
def run_crm_from_predictions(cfg: CRMConfig) -> Tuple[CRMSimulationResults, pd.DataFrame]:
    ref_df = load_predictions(cfg)
    exp_df = load_experiment(cfg)

    t_ref = ref_df[cfg.time_col].to_numpy(float)
    X0 = float(ref_df[cfg.biomass_col].iloc[0])
    G0 = float(ref_df[cfg.glucose_col].iloc[0])
    A0 = float(ref_df[cfg.acetate_col].iloc[0])
    O0 = float(ref_df[cfg.oxygen_col].iloc[0])

    end_time = float(t_ref[-1]) if cfg.end_time_hours is None else float(cfg.end_time_hours)

    if cfg.use_phase_based_crm:
        p1, p2 = extract_phase_parameters(ref_df, cfg)
        print_params("Phase 1 CRM parameters", p1)
        print_params("Phase 2 CRM parameters", p2)

        crm = simulate_crm_phase_based(
            p1, p2,
            X0=X0, G0=G0, A0=A0, O0=O0,
            end_time_hours=end_time,
            step_hours=cfg.step_hours,
            threshold=cfg.glucose_phase_threshold,
        )
    else:
        p = extract_constant_parameters(ref_df, cfg)
        print_params("Constant CRM parameters", p)

        crm = simulate_crm_constant(
            p,
            X0=X0, G0=G0, A0=A0, O0=O0,
            end_time_hours=end_time,
            step_hours=cfg.step_hours,
        )

    # compare
    biomass_ref_on_crm = np.interp(crm.time, t_ref, ref_df[cfg.biomass_col].to_numpy(float))
    glucose_ref_on_crm = np.interp(crm.time, t_ref, ref_df[cfg.glucose_col].to_numpy(float))
    acetate_ref_on_crm = np.interp(crm.time, t_ref, ref_df[cfg.acetate_col].to_numpy(float))
    oxygen_ref_on_crm = np.interp(crm.time, t_ref, ref_df[cfg.oxygen_col].to_numpy(float))

    print("\n=== CRM vs reference NMSE ===")
    print(f"Biomass NMSE : {normalized_mse(crm.biomass, biomass_ref_on_crm, scale=1.0):.6f}")
    print(f"Glucose NMSE : {normalized_mse(crm.glucose, glucose_ref_on_crm, scale=20.0):.6f}")
    print(f"Acetate NMSE : {normalized_mse(crm.acetate, acetate_ref_on_crm, scale=20.0):.6f}")
    print(f"Oxygen NMSE  : {normalized_mse(crm.oxygen, oxygen_ref_on_crm, scale=20.0):.6f}")

    crm_df = pd.DataFrame({
        "time": crm.time,
        "biomass": crm.biomass,
        "glucose": crm.glucose,
        "acetate": crm.acetate,
        "oxygen": crm.oxygen,
        "mu": crm.mu,
        "ug": crm.ug,
        "ua": crm.ua,
        "uo": crm.uo,
        "phase": crm.phase,
    })
    out_csv = "crm_simulation_from_predictions.csv"
    crm_df.to_csv(out_csv, index=False)
    print(f"\nSaved CRM simulation to: {out_csv}")

    if cfg.plot:
        plot_results(ref_df, crm, cfg, exp_df=exp_df)

    return crm, crm_df


# ============================================================
# Main
# ============================================================
def main():
    cfg = CRMConfig(
        predictions_csv="predictions.csv",
        experiment_csv="/Users/edwin/Downloads/plot-data (2).csv",

        # choose:
        use_phase_based_crm=False,

        # phase split threshold
        glucose_phase_threshold=0.5,

        # simulation resolution
        step_hours=0.01,

        plot=True,
    )

    run_crm_from_predictions(cfg)


if __name__ == "__main__":
    main()