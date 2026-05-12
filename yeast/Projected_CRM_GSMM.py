from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from cobra.io import read_sbml_model
from cobra.util.solver import check_solver_status

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ============================================================
# Config
# ============================================================
@dataclass
class BiomassProjectedSubstrateMatchedConfig:
    csv_path: str
    model_path: str

    # CSV columns
    time_col: str = "time"
    density_col: str = "cells_per_ml"
    gal_col: str = "galactose"
    etoh_col: str = "ethanol"

    # GSMM reaction IDs
    biomass_rxn_id: str = "BIOMASS_SC5_notrace"
    gal_rxn_id: str = "EX_gal_e"
    etoh_rxn_id: str = "EX_etoh_e"

    # unit conversions
    gDW_per_cell: float = 2e-12
    gal_concentration_to_mmol_L: float = 1000.0
    etoh_concentration_to_mmol_L: float = 1000.0

    # smoothing
    density_sigma: float = 2.0
    gal_sigma: float = 2.0
    etoh_sigma: float = 2.0

    # numerical guards
    min_density_cells_ml: float = 1e-12
    min_biomass_gDW_L: float = 1e-15
    min_dt: float = 1e-12
    eps: float = 1e-30

    # initialization
    use_crm_initial_state: bool = True

    # Stage 1: biomass projection
    weight_mu: float = 1.0
    use_mu_window: bool = False
    mu_rel_window: float = 0.50
    mu_abs_window: float = 1e-6

    # Stage 2: biomass fixing for substrate matching
    mu_fix_rel_tol: float = 1e-6
    mu_fix_abs_tol: float = 1e-8

    # Stage 2: substrate matching weights
    weight_gal_match: float = 10.0
    weight_etoh_match: float = 5.0

    # Optional local windows around substrate targets in stage 2
    use_substrate_windows: bool = False
    gal_rel_window: float = 0.30
    etoh_rel_window: float = 0.30
    substrate_abs_window: float = 1e-6

    # medium defaults
    gal_medium_bound: float = 10
    etoh_medium_bound: float = 5

    # fallback
    allow_rescue_with_targets: bool = True

    # state propagation
    clamp_concentrations_nonnegative: bool = True

    output_dir: str = "biomass_projected_substrate_matched_run"


# ============================================================
# Helpers
# ============================================================
def compute_mu_from_density_smoothed(
    time: np.ndarray,
    density_cells_ml: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    density_cells_ml = np.maximum(density_cells_ml, 1e-12)
    logX = np.log(density_cells_ml)
    logX_smooth = gaussian_filter1d(logX, sigma=sigma)
    density_smooth = np.exp(logX_smooth)

    dt = np.diff(time)
    mu_fwd = np.diff(logX_smooth) / dt
    mu = np.append(mu_fwd, mu_fwd[-1])

    return density_smooth, mu


def convert_cellsml_to_gdwl(cells_ml: np.ndarray, gDW_per_cell: float) -> np.ndarray:
    return np.maximum(cells_ml * gDW_per_cell * 1000.0, 0.0)


def convert_gdwl_to_cellsml(gdwl: np.ndarray, gDW_per_cell: float) -> np.ndarray:
    return gdwl / (gDW_per_cell * 1000.0)


def interval_biomass_integral(Xk: float, mu_k: float, dt: float) -> float:
    if dt <= 0:
        return 0.0
    if abs(mu_k) < 1e-12:
        return Xk * dt
    return Xk * (np.exp(mu_k * dt) - 1.0) / mu_k


def build_window(center: float, rel_window: float, abs_window: float) -> tuple[float, float]:
    width = max(abs(center) * rel_window, abs_window)
    return center - width, center + width


def build_fix_bounds(value: float, rel_tol: float, abs_tol: float) -> tuple[float, float]:
    width = max(abs(value) * rel_tol, abs_tol)
    return value - width, value + width


# ============================================================
# 1. Preprocess CRM data
# ============================================================
def preprocess_crm_data(cfg: BiomassProjectedSubstrateMatchedConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    needed = [cfg.time_col, cfg.density_col, cfg.gal_col, cfg.etoh_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed).sort_values(cfg.time_col).reset_index(drop=True)

    time = df[cfg.time_col].to_numpy(float)
    density_raw = np.maximum(df[cfg.density_col].to_numpy(float), cfg.min_density_cells_ml)
    gal_raw = df[cfg.gal_col].to_numpy(float) * cfg.gal_concentration_to_mmol_L
    etoh_raw = df[cfg.etoh_col].to_numpy(float) * cfg.etoh_concentration_to_mmol_L

    density_smooth, mu_crm = compute_mu_from_density_smoothed(
        time=time,
        density_cells_ml=density_raw,
        sigma=cfg.density_sigma,
    )
    gal_smooth = gaussian_filter1d(gal_raw, sigma=cfg.gal_sigma)
    etoh_smooth = gaussian_filter1d(etoh_raw, sigma=cfg.etoh_sigma)

    biomass_gDW_L = convert_cellsml_to_gdwl(density_smooth, cfg.gDW_per_cell)
    biomass_gDW_L = np.maximum(biomass_gDW_L, cfg.min_biomass_gDW_L)

    out = pd.DataFrame({
        "Time": time,
        "density_raw_cells_ml": density_raw,
        "density_smooth_cells_ml": density_smooth,
        "biomass_gDW_L": biomass_gDW_L,
        "mu_crm": mu_crm,
        "gal_raw_mM": gal_raw,
        "gal_smooth_mM": gal_smooth,
        "etoh_raw_mM": etoh_raw,
        "etoh_smooth_mM": etoh_smooth,
    })
    return out


# ============================================================
# 2. Infer interval targets from CRM
# ============================================================
def infer_target_fluxes(
    df: pd.DataFrame,
    cfg: BiomassProjectedSubstrateMatchedConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = df["Time"].to_numpy(float)
    X = df["biomass_gDW_L"].to_numpy(float)
    mu = df["mu_crm"].to_numpy(float)
    S_g = df["gal_smooth_mM"].to_numpy(float)
    S_e = df["etoh_smooth_mM"].to_numpy(float)

    rows = []
    for k in range(len(df) - 1):
        dt = max(t[k + 1] - t[k], cfg.min_dt)
        bm_int = interval_biomass_integral(X[k], mu[k], dt)

        dS_g = S_g[k + 1] - S_g[k]
        dS_e = S_e[k + 1] - S_e[k]

        q_gal = -dS_g / max(bm_int, cfg.eps)
        q_etoh = -dS_e / max(bm_int, cfg.eps)

        rows.append({
            "interval_index": k,
            "t_start": t[k],
            "t_end": t[k + 1],
            "dt_h": dt,
            "X_start_gDW_L": X[k],
            "mu_target": mu[k],
            "gal_start_mM": S_g[k],
            "gal_end_mM": S_g[k + 1],
            "etoh_start_mM": S_e[k],
            "etoh_end_mM": S_e[k + 1],
            "q_gal_target": q_gal,
            "q_etoh_target": q_etoh,
            "v_gal_target": -q_gal,
            "v_etoh_target": -q_etoh,
        })

    flux_df = pd.DataFrame(rows)

    point_df = df.copy()
    point_df["mu_target_interval"] = np.nan
    point_df["v_gal_target"] = np.nan
    point_df["v_etoh_target"] = np.nan
    point_df.iloc[:-1, point_df.columns.get_loc("mu_target_interval")] = flux_df["mu_target"].to_numpy()
    point_df.iloc[:-1, point_df.columns.get_loc("v_gal_target")] = flux_df["v_gal_target"].to_numpy()
    point_df.iloc[:-1, point_df.columns.get_loc("v_etoh_target")] = flux_df["v_etoh_target"].to_numpy()

    return point_df, flux_df


# ============================================================
# 3. Prepare model
# ============================================================
def prepare_yeast_model(cfg: BiomassProjectedSubstrateMatchedConfig):
    model = read_sbml_model(cfg.model_path)

    # Remove glucose via medium setter (this one works)
    media = model.medium.copy()
    media.pop("EX_glc__D_e", None)
    model.medium = media

    # Set galactose and ethanol bounds directly —
    # model.medium silently fails for these reactions in iMM904
    gal_rxn  = model.reactions.get_by_id(cfg.gal_rxn_id)
    etoh_rxn = model.reactions.get_by_id(cfg.etoh_rxn_id)

    # Uptake is negative flux in COBRA convention
    gal_rxn.lower_bound  = -cfg.gal_medium_bound
    etoh_rxn.lower_bound = -cfg.etoh_medium_bound

    # Verify
    print(f"EX_gal_e  lb={gal_rxn.lower_bound}  ub={gal_rxn.upper_bound}")
    print(f"EX_etoh_e lb={etoh_rxn.lower_bound}  ub={etoh_rxn.upper_bound}")

    return model


# ============================================================
# 4. Stage 1: project only biomass
# ============================================================
def project_biomass_target_to_feasible(
    model,
    mu_target: float,
    cfg: BiomassProjectedSubstrateMatchedConfig,
) -> dict:
    with model as m:
        bio = m.reactions.get_by_id(cfg.biomass_rxn_id)
        prob = m.problem

        if cfg.use_mu_window:
            mu_lb, mu_ub = build_window(mu_target, cfg.mu_rel_window, cfg.mu_abs_window)
            bio.lower_bound = max(0.0, mu_lb)
            bio.upper_bound = max(0.0, mu_ub)
        else:
            bio.lower_bound = max(0.0, bio.lower_bound)

        mu_pos = prob.Variable("mu_pos_dev", lb=0.0)
        mu_neg = prob.Variable("mu_neg_dev", lb=0.0)
        m.add_cons_vars([mu_pos, mu_neg])

        c_mu = prob.Constraint(
            bio.flux_expression - mu_target - mu_pos + mu_neg,
            lb=0.0, ub=0.0, name="mu_projection_balance"
        )
        m.add_cons_vars([c_mu])

        deviation_expr = cfg.weight_mu * (mu_pos + mu_neg)
        m.objective = prob.Objective(deviation_expr, direction="min")

        sol = m.optimize()
        check_solver_status(m.solver.status)

        mu_proj = float(sol.fluxes[cfg.biomass_rxn_id])

        return {
            "status": sol.status,
            "projection_objective": float(sol.objective_value),
            "mu_projected": mu_proj,
            "delta_mu": mu_proj - mu_target,
        }


# ============================================================
# 5. Stage 2: at matched biomass, minimize substrate mismatch
# ============================================================
def realize_fluxes_at_projected_biomass_with_substrate_matching(
    model,
    mu_projected: float,
    v_gal_target: float,
    v_etoh_target: float,
    cfg: BiomassProjectedSubstrateMatchedConfig,
) -> dict:
    with model as m:
        bio = m.reactions.get_by_id(cfg.biomass_rxn_id)
        gal = m.reactions.get_by_id(cfg.gal_rxn_id)
        etoh = m.reactions.get_by_id(cfg.etoh_rxn_id)

        prob = m.problem

        # Fix biomass tightly around projected value
        mu_lb, mu_ub = build_fix_bounds(
            value=mu_projected,
            rel_tol=cfg.mu_fix_rel_tol,
            abs_tol=cfg.mu_fix_abs_tol,
        )
        bio.lower_bound = max(0.0, mu_lb)
        bio.upper_bound = max(0.0, mu_ub)

        # Optional local windows on substrate targets
        if cfg.use_substrate_windows:
            gal_lb, gal_ub = build_window(v_gal_target, cfg.gal_rel_window, cfg.substrate_abs_window)
            etoh_lb, etoh_ub = build_window(v_etoh_target, cfg.etoh_rel_window, cfg.substrate_abs_window)
            gal.lower_bound = max(gal.lower_bound, gal_lb)
            gal.upper_bound = min(gal.upper_bound, gal_ub)
            etoh.lower_bound = max(etoh.lower_bound, etoh_lb)
            etoh.upper_bound = min(etoh.upper_bound, etoh_ub)

        # Absolute deviation variables for galactose
        gal_pos = prob.Variable("gal_pos_dev", lb=0.0)
        gal_neg = prob.Variable("gal_neg_dev", lb=0.0)

        # Absolute deviation variables for ethanol
        etoh_pos = prob.Variable("etoh_pos_dev", lb=0.0)
        etoh_neg = prob.Variable("etoh_neg_dev", lb=0.0)

        m.add_cons_vars([gal_pos, gal_neg, etoh_pos, etoh_neg])

        c_gal = prob.Constraint(
            gal.flux_expression - v_gal_target - gal_pos + gal_neg,
            lb=0.0, ub=0.0, name="gal_target_balance"
        )
        c_etoh = prob.Constraint(
            etoh.flux_expression - v_etoh_target - etoh_pos + etoh_neg,
            lb=0.0, ub=0.0, name="etoh_target_balance"
        )
        m.add_cons_vars([c_gal, c_etoh])

        obj = (
            cfg.weight_gal_match * (gal_pos + gal_neg) +
            cfg.weight_etoh_match * (etoh_pos + etoh_neg)
        )
        m.objective = prob.Objective(obj, direction="min")

        sol = m.optimize()
        check_solver_status(m.solver.status)

        mu_realized = float(sol.fluxes[cfg.biomass_rxn_id])
        v_gal_realized = float(sol.fluxes[cfg.gal_rxn_id])
        v_etoh_realized = float(sol.fluxes[cfg.etoh_rxn_id])

        return {
            "status": sol.status,
            "match_objective": float(sol.objective_value),
            "mu_realized": mu_realized,
            "v_gal_realized": v_gal_realized,
            "v_etoh_realized": v_etoh_realized,
            "q_gal_realized": -v_gal_realized,
            "q_etoh_realized": -v_etoh_realized,
            "delta_v_gal_vs_target": v_gal_realized - v_gal_target,
            "delta_v_etoh_vs_target": v_etoh_realized - v_etoh_target,
        }


# ============================================================
# 6. Forward simulation
# ============================================================
def simulate_biomass_projected_substrate_matched_tracking(
    model,
    flux_df: pd.DataFrame,
    point_df: pd.DataFrame,
    cfg: BiomassProjectedSubstrateMatchedConfig,
) -> pd.DataFrame:
    if cfg.use_crm_initial_state:
        X_curr = float(point_df["biomass_gDW_L"].iloc[0])
        gal_curr = float(point_df["gal_smooth_mM"].iloc[0])
        etoh_curr = float(point_df["etoh_smooth_mM"].iloc[0])
    else:
        X_curr = max(cfg.min_biomass_gDW_L, 1e-6)
        gal_curr = float(point_df["gal_smooth_mM"].iloc[0])
        etoh_curr = float(point_df["etoh_smooth_mM"].iloc[0])

    rows = []
    first_failure = None
    failure_rows = []

    for _, row in flux_df.iterrows():
        interval_index = int(row["interval_index"])
        t0 = float(row["t_start"])
        t1 = float(row["t_end"])
        dt = float(row["dt_h"])

        mu_target = float(row["mu_target"])
        v_gal_target = float(row["v_gal_target"])
        v_etoh_target = float(row["v_etoh_target"])

        status = "optimal"

        try:
            proj = project_biomass_target_to_feasible(
                model=model,
                mu_target=mu_target,
                cfg=cfg,
            )

            realized = realize_fluxes_at_projected_biomass_with_substrate_matching(
                model=model,
                mu_projected=proj["mu_projected"],
                v_gal_target=v_gal_target,
                v_etoh_target=v_etoh_target,
                cfg=cfg,
            )

            mu_exec = realized["mu_realized"]
            v_gal_exec = realized["v_gal_realized"]
            v_etoh_exec = realized["v_etoh_realized"]

        except Exception as exc:
            status = "biomass_projection_or_substrate_match_failed_rescue"
            err_msg = str(exc)

            proj = {
                "status": status,
                "projection_objective": np.nan,
                "mu_projected": np.nan,
                "delta_mu": np.nan,
            }

            realized = {
                "status": status,
                "match_objective": np.nan,
                "mu_realized": np.nan,
                "v_gal_realized": np.nan,
                "v_etoh_realized": np.nan,
                "q_gal_realized": np.nan,
                "q_etoh_realized": np.nan,
                "delta_v_gal_vs_target": np.nan,
                "delta_v_etoh_vs_target": np.nan,
            }

            if cfg.allow_rescue_with_targets:
                mu_exec = mu_target
                v_gal_exec = v_gal_target
                v_etoh_exec = v_etoh_target
            else:
                raise RuntimeError(f"Failure at interval {interval_index} (t={t0}): {err_msg}")

            fail_info = {
                "interval_index": interval_index,
                "t_start": t0,
                "t_end": t1,
                "status": status,
                "error_message": err_msg,
                "mu_target": mu_target,
                "v_gal_target": v_gal_target,
                "v_etoh_target": v_etoh_target,
            }
            failure_rows.append(fail_info)

            if first_failure is None:
                first_failure = fail_info
                logger.warning(
                    "First failure begins at t=%.4f h (interval %d, ends %.4f h) | error=%s",
                    t0, interval_index, t1, err_msg
                )

        bm_int = interval_biomass_integral(X_curr, mu_exec, dt)

        X_next = X_curr * np.exp(mu_exec * dt)
        gal_next = gal_curr + v_gal_exec * bm_int
        etoh_next = etoh_curr + v_etoh_exec * bm_int

        if cfg.clamp_concentrations_nonnegative:
            gal_next = max(gal_next, 0.0)
            etoh_next = max(etoh_next, 0.0)

        rows.append({
            "interval_index": interval_index,
            "Time": t0,
            "t_end": t1,
            "dt_h": dt,
            "status": status if status != "ok" else f"{proj['status']}|{realized['status']}",

            "X_pred_start_gDW_L": X_curr,
            "gal_pred_start_mM": gal_curr,
            "etoh_pred_start_mM": etoh_curr,

            # CRM targets
            "mu_target": mu_target,
            "v_gal_target": v_gal_target,
            "v_etoh_target": v_etoh_target,
            "q_gal_target": -v_gal_target,
            "q_etoh_target": -v_etoh_target,

            # Stage 1 output
            "projection_objective": proj["projection_objective"],
            "mu_projected": proj["mu_projected"],
            "delta_mu": proj["delta_mu"],

            # Stage 2 output
            "match_objective": realized["match_objective"],
            "mu_realized": realized["mu_realized"],
            "v_gal_realized": realized["v_gal_realized"],
            "v_etoh_realized": realized["v_etoh_realized"],
            "q_gal_realized": realized["q_gal_realized"],
            "q_etoh_realized": realized["q_etoh_realized"],
            "delta_v_gal_vs_target": realized["delta_v_gal_vs_target"],
            "delta_v_etoh_vs_target": realized["delta_v_etoh_vs_target"],

            # Forward state
            "X_pred_next_gDW_L": X_next,
            "gal_pred_next_mM": gal_next,
            "etoh_pred_next_mM": etoh_next,
        })

        X_curr = max(X_next, cfg.min_biomass_gDW_L)
        gal_curr = gal_next
        etoh_curr = etoh_next

    sim_df = pd.DataFrame(rows)

    os.makedirs(cfg.output_dir, exist_ok=True)
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(
            os.path.join(cfg.output_dir, "biomass_projected_substrate_matched_failures.csv"),
            index=False
        )
        print("\nFIRST FAILURE:")
        print(first_failure)
    else:
        print("\nNo failures found.")

    sim_df["density_predicted_cells_ml"] = convert_gdwl_to_cellsml(
        sim_df["X_pred_start_gDW_L"].to_numpy(float),
        cfg.gDW_per_cell,
    )
    sim_df["gal_predicted_mM"] = sim_df["gal_pred_start_mM"]
    sim_df["etoh_predicted_mM"] = sim_df["etoh_pred_start_mM"]

    return sim_df


# ============================================================
# 7. Plotting
# ============================================================
def plot_outputs(
    pre_df: pd.DataFrame,
    flux_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    cfg: BiomassProjectedSubstrateMatchedConfig,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    t_raw = pre_df["Time"].to_numpy()
    t_sim = sim_df["Time"].to_numpy()

    # Biomass
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t_raw, pre_df["density_raw_cells_ml"], lw=1.2, alpha=0.4, color="tab:gray", label="Raw (CRM)")
    axes[0].plot(t_raw, pre_df["density_smooth_cells_ml"], lw=2.0, color="tab:blue", label="Smoothed (CRM)")
    axes[0].plot(t_sim, sim_df["density_predicted_cells_ml"], lw=2.0, ls="--", color="tab:red",
                 label="Biomass-projected + substrate-matched GSMM tracking")
    axes[0].set_ylabel("Cells / mL")
    axes[0].set_title("Biomass: CRM vs biomass-projected + substrate-matched GSMM tracking")
    axes[0].legend()

    crm_interp = np.interp(t_sim, t_raw, pre_df["density_smooth_cells_ml"].to_numpy())
    residual = sim_df["density_predicted_cells_ml"].to_numpy() - crm_interp
    axes[1].plot(t_sim, residual, lw=1.5, color="tab:red")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Time (h)")
    axes[1].set_title("Residual (GSMM biomass - smoothed CRM biomass)")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "biomass_tracking.png"), dpi=300)
    plt.close(fig)

    # Galactose
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_raw, pre_df["gal_raw_mM"], lw=1.2, alpha=0.4, color="tab:gray", label="Raw (CRM)")
    ax.plot(t_raw, pre_df["gal_smooth_mM"], lw=2.0, color="tab:green", label="Smoothed (CRM)")
    ax.plot(t_sim, sim_df["gal_predicted_mM"], lw=2.0, ls="--", color="tab:red",
            label="Biomass-projected + substrate-matched GSMM tracking")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Galactose (mmol/L)")
    ax.set_title("Galactose: CRM vs biomass-projected + substrate-matched GSMM tracking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "galactose_tracking.png"), dpi=300)
    plt.close(fig)

    # Ethanol
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_raw, pre_df["etoh_raw_mM"], lw=1.2, alpha=0.4, color="tab:gray", label="Raw (CRM)")
    ax.plot(t_raw, pre_df["etoh_smooth_mM"], lw=2.0, color="tab:orange", label="Smoothed (CRM)")
    ax.plot(t_sim, sim_df["etoh_predicted_mM"], lw=2.0, ls="--", color="tab:red",
            label="Biomass-projected + substrate-matched GSMM tracking")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Ethanol (mmol/L)")
    ax.set_title("Ethanol: CRM vs biomass-projected + substrate-matched GSMM tracking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "ethanol_tracking.png"), dpi=300)
    plt.close(fig)

    # Targets vs realized
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t_sim, flux_df["mu_target"], lw=2.0, color="tab:blue", label="CRM target")
    axes[0].plot(t_sim, sim_df["mu_projected"], lw=2.0, ls="--", color="tab:red", label="Projected feasible biomass")
    axes[0].plot(t_sim, sim_df["mu_realized"], lw=1.8, ls=":", color="tab:purple", label="Realized GSMM biomass")
    axes[0].set_ylabel("1/h")
    axes[0].set_title("Growth-rate target vs projected/realized biomass")
    axes[0].legend()

    axes[1].plot(t_sim, flux_df["q_gal_target"], lw=2.0, color="tab:green", label="CRM target uptake")
    axes[1].plot(t_sim, sim_df["q_gal_realized"], lw=2.0, ls="--", color="tab:red", label="GSMM realized uptake")
    axes[1].set_ylabel("mmol / gDW / h")
    axes[1].set_title("Galactose uptake target vs GSMM realized uptake")
    axes[1].legend()

    axes[2].plot(t_sim, flux_df["v_etoh_target"], lw=2.0, color="tab:orange", label="CRM target exchange")
    axes[2].plot(t_sim, sim_df["v_etoh_realized"], lw=2.0, ls="--", color="tab:red", label="GSMM realized exchange")
    axes[2].axhline(0, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("Time (h)")
    axes[2].set_ylabel("mmol / gDW / h")
    axes[2].set_title("Ethanol exchange target vs GSMM realized exchange")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "targets_vs_realized.png"), dpi=300)
    plt.close(fig)

    # Deltas
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t_sim, sim_df["delta_mu"], lw=2.0)
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("Δμ")
    axes[0].set_title("Biomass projection correction")

    axes[1].plot(t_sim, sim_df["delta_v_gal_vs_target"], lw=2.0)
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Δv_gal")
    axes[1].set_title("Galactose exchange disagreement at matched biomass")

    axes[2].plot(t_sim, sim_df["delta_v_etoh_vs_target"], lw=2.0)
    axes[2].axhline(0, color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("Δv_etoh")
    axes[2].set_xlabel("Time (h)")
    axes[2].set_title("Ethanol exchange disagreement at matched biomass")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "deltas.png"), dpi=300)
    plt.close(fig)

    pre_df.to_csv(os.path.join(cfg.output_dir, "preprocessed_crm.csv"), index=False)
    flux_df.to_csv(os.path.join(cfg.output_dir, "crm_targets.csv"), index=False)
    sim_df.to_csv(os.path.join(cfg.output_dir, "tracking_predictions.csv"), index=False)

    logger.info("Saved results to: %s", cfg.output_dir)


# ============================================================
# 8. Main
# ============================================================
def run_pipeline(cfg: BiomassProjectedSubstrateMatchedConfig) -> None:
    logger.info("Stage 1: preprocess CRM")
    pre_df = preprocess_crm_data(cfg)

    logger.info("Stage 2: infer CRM interval targets")
    point_df, flux_df = infer_target_fluxes(pre_df, cfg)

    logger.info("Stage 3: load GSMM")
    model = prepare_yeast_model(cfg)

    logger.info("Stage 4: biomass projection + substrate matching + forward simulation")
    sim_df = simulate_biomass_projected_substrate_matched_tracking(model, flux_df, point_df, cfg)

    logger.info("Stage 5: plot and save")
    plot_outputs(pre_df, flux_df, sim_df, cfg)

    good = ~sim_df["status"].str.contains("failed", case=False, na=False)
    if good.any():
        logger.info("Mean |delta_mu| = %.6f", np.nanmean(np.abs(sim_df.loc[good, "delta_mu"])))
        logger.info("Mean |delta_v_gal_vs_target| = %.6f",
                    np.nanmean(np.abs(sim_df.loc[good, "delta_v_gal_vs_target"])))
        logger.info("Mean |delta_v_etoh_vs_target| = %.6f",
                    np.nanmean(np.abs(sim_df.loc[good, "delta_v_etoh_vs_target"])))

    print(sim_df.head(20).to_string(index=False))


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    cfg = BiomassProjectedSubstrateMatchedConfig(
        csv_path="adaptive_diauxic_shift.csv",
        model_path="/Users/edwin/Downloads/iMM904.xml",

        time_col="time",
        density_col="cells_per_ml",
        gal_col="galactose",
        etoh_col="ethanol",

        biomass_rxn_id="BIOMASS_SC5_notrace",
        gal_rxn_id="EX_gal_e",
        etoh_rxn_id="EX_etoh_e",

        gDW_per_cell=2e-12,
        density_sigma=2.0,
        gal_sigma=2.0,
        etoh_sigma=2.0,

        gal_concentration_to_mmol_L=1000.0,
        etoh_concentration_to_mmol_L=1000.0,

        # Stage 1: biomass projection
        weight_mu=1.0,
        use_mu_window=True,
        mu_rel_window=0.20,
        mu_abs_window=1e-6,

        # Stage 2: biomass fixed tightly
        mu_fix_rel_tol=1e-6,
        mu_fix_abs_tol=1e-8,

        gal_medium_bound = 100,
        etoh_medium_bound = 100,

        # Stage 2: substrate matching weights
        weight_gal_match=10.0,
        weight_etoh_match=5.0,

        # Optional local windows
        use_substrate_windows=True,
        gal_rel_window=0.20,
        etoh_rel_window=0.50,
        substrate_abs_window=1e-6,

        allow_rescue_with_targets=False,
        output_dir="biomass_projected_substrate_matched_run",
    )

    run_pipeline(cfg)