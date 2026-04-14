"""
crm_mm_dfba_infer_yields.py

Michaelis-Menten uptake from CRM, effective yields inferred from dFBA.

What this script does
---------------------
1. Uses CRM Michaelis-Menten uptake laws to compute:
      ug_crm, ua_crm, uo_crm

2. Uses those as dFBA exchange bounds:
      EX_glucose >= -ug_crm
      EX_acetate >= -ua_crm
      EX_oxygen  >= -uo_crm

3. Solves dFBA at each time step

4. Infers effective reduced-model traits from the realized dFBA solution:
      Yg(t)    ~ mu / (ug * oxygen_effect)   during glucose-dominant phase
      Ya(t)    ~ mu / (ua * oxygen_effect)   during acetate-dominant phase
      alpha(t) ~ acetate_secretion / glucose_uptake

5. Simulates a CRM ODE using:
      - MM uptake from CRM kinetics
      - time-varying inferred yields/overflow from dFBA

This gives you:
- dFBA trajectory
- inferred reduced traits from dFBA
- CRM trajectory driven by those inferred traits
- plots vs experiment

Important note
--------------
The inferred yields do NOT go directly into the dFBA optimization.
They are inferred FROM dFBA and then used by the CRM layer.

Dependencies
------------
pip install cobra numpy pandas matplotlib scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

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
class CRMKinetics:
    # Michaelis-Menten uptake capacities
    Vg: float = 11.5
    Va: float = 10.0
    Vo: float = 13.8

    # Michaelis-Menten constants
    Kg: float = 0.05
    Ka: float = 0.05
    Ko: float = 0.05

    # Maintenance stays a CRM-side term
    maintenance: float = 0.0

    # Optional acetate gate
    enable_acetate_gate: bool = True
    glucose_gate_threshold: float = 0.5

    # Fallback / initial reduced parameters before dFBA informs them
    Yg_init: float = 0.08
    Ya_init: float = 0.03
    alpha_init: float = 0.5


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
    end_time_hours: float = 10.0

    initial_biomass: float = 0.0045
    initial_glucose: float = 11.1
    initial_acetate: float = 0.0
    initial_oxygen: float = 50.0

    use_availability_caps: bool = True

    # inference settings
    uptake_eps: float = 1e-8
    phase_tol: float = 1e-4
    ema: float = 0.15  # smoothing for inferred traits

    experiment_csv: Optional[str] = "/Users/edwin/Downloads/plot-data (2).csv"
    exp_time_col: str = "Time"
    exp_biomass_col: str = "Biomass"

    output_dfba_csv: str = "dfba_with_inferred_traits.csv"
    output_crm_csv: str = "crm_from_inferred_traits.csv"
    plot: bool = True

    def __post_init__(self):
        if self.byproduct_exchanges is None:
            self.byproduct_exchanges = [
                "EX_ac_e", "EX_etoh_e", "EX_for_e", "EX_lac__D_e", "EX_succ_e"
            ]
        if self.base_medium_exchanges is None:
            self.base_medium_exchanges = [
                "EX_h_e", "EX_h2o_e", "EX_na1_e", "EX_k_e", "EX_pi_e",
                "EX_so4_e", "EX_nh4_e", "EX_cl_e", "EX_mg2_e", "EX_ca2_e",
            ]


# ============================================================
# Helpers
# ============================================================
def michaelis_menten(S: float, Vmax: float, Km: float) -> float:
    S = max(0.0, float(S))
    return Vmax * S / (Km + S + 1e-12)


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


def availability_cap_mmol_per_gdw_hr(concentration: float, biomass: float, dt: float) -> float:
    if biomass <= 0.0 or dt <= 0.0:
        return 0.0
    return max(0.0, float(concentration) / (float(biomass) * float(dt)))


def update_concentration_eq7(
    concentration: float,
    exchange_flux: float,
    biomass: float,
    mu: float,
    dt: float
) -> float:
    substrate_uptake = -float(exchange_flux)

    if abs(mu) < 1e-12:
        return float(concentration) - substrate_uptake * float(biomass) * float(dt)

    return float(concentration) + (
        substrate_uptake * float(biomass) / float(mu)
    ) * (1.0 - np.exp(float(mu) * float(dt)))


def normalized_mse(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(((a - b) / float(scale)) ** 2))


# ============================================================
# CRM MM uptake
# ============================================================
def compute_crm_uptakes(
    glucose: float,
    acetate: float,
    oxygen: float,
    kinetics: CRMKinetics,
) -> Dict[str, float]:
    ug = michaelis_menten(glucose, kinetics.Vg, kinetics.Kg)
    ua = michaelis_menten(acetate, kinetics.Va, kinetics.Ka)
    uo = michaelis_menten(oxygen, kinetics.Vo, kinetics.Ko)

    if kinetics.enable_acetate_gate and glucose > kinetics.glucose_gate_threshold:
        ua = 0.0

    return {
        "ug_crm": ug,
        "ua_crm": ua,
        "uo_crm": uo,
    }


# ============================================================
# dFBA + trait inference
# ============================================================
def run_crm_guided_mm_dfba_and_infer_traits(
    kinetics: CRMKinetics,
    cfg: DFBAConfig,
) -> pd.DataFrame:
    model = load_model(cfg.model_name)
    validate_ids(model, cfg)

    time_grid = np.arange(0.0, cfg.end_time_hours + 1e-12, cfg.step_hours, dtype=float)

    biomass = float(cfg.initial_biomass)
    glucose = float(cfg.initial_glucose)
    acetate = float(cfg.initial_acetate)
    oxygen = float(cfg.initial_oxygen)

    # running inferred traits
    Yg_hat = float(kinetics.Yg_init)
    Ya_hat = float(kinetics.Ya_init)
    alpha_hat = float(kinetics.alpha_init)

    rows = []

    for t in time_grid:
        crm_rates = compute_crm_uptakes(glucose, acetate, oxygen, kinetics)
        ug_crm = crm_rates["ug_crm"]
        ua_crm = crm_rates["ua_crm"]
        uo_crm = crm_rates["uo_crm"]

        if cfg.use_availability_caps:
            ug_cap = availability_cap_mmol_per_gdw_hr(glucose, biomass, cfg.step_hours)
            ua_cap = availability_cap_mmol_per_gdw_hr(acetate, biomass, cfg.step_hours)
            uo_cap = availability_cap_mmol_per_gdw_hr(oxygen, biomass, cfg.step_hours)

            ug_bound = min(ug_crm, ug_cap)
            ua_bound = min(ua_crm, ua_cap)
            uo_bound = min(uo_crm, uo_cap)
        else:
            ug_bound = ug_crm
            ua_bound = ua_crm
            uo_bound = uo_crm

        row = {
            "time": t,
            "biomass": biomass,
            "glucose": glucose,
            "acetate": acetate,
            "oxygen": oxygen,
            "ug_crm": ug_crm,
            "ua_crm": ua_crm,
            "uo_crm": uo_crm,
            "ug_bound": ug_bound,
            "ua_bound": ua_bound,
            "uo_bound": uo_bound,
            "mu_dfba": np.nan,
            "v_glucose": np.nan,
            "v_acetate": np.nan,
            "v_oxygen": np.nan,
            "ug_dfba": np.nan,
            "ua_dfba": np.nan,
            "uo_dfba": np.nan,
            "Yg_inferred": Yg_hat,
            "Ya_inferred": Ya_hat,
            "alpha_inferred": alpha_hat,
            "oxygen_effect": np.nan,
        }

        if t >= cfg.end_time_hours:
            rows.append(row)
            break

        with model as m:
            m.objective = cfg.biomass_reaction
            open_inorganic_base_medium(m, cfg)

            for rid in cfg.byproduct_exchanges:
                if rid in m.reactions:
                    r = m.reactions.get_by_id(rid)
                    r.lower_bound = 0.0
                    r.upper_bound = 1000.0

            m.reactions.get_by_id(cfg.ex_glucose).lower_bound = -abs(ug_bound)
            m.reactions.get_by_id(cfg.ex_acetate).lower_bound = -abs(ua_bound)
            m.reactions.get_by_id(cfg.ex_oxygen).lower_bound = -abs(uo_bound)

            sol = solve_fba(m, cfg.use_pfba)
            if sol.status != "optimal":
                print(f"Stopping at t={t:.3f} h because solver status was {sol.status}")
                rows.append(row)
                break

            mu = float(sol.objective_value)
            v_glucose = float(sol.fluxes[cfg.ex_glucose])
            v_acetate = float(sol.fluxes[cfg.ex_acetate])
            v_oxygen = float(sol.fluxes[cfg.ex_oxygen])

        ug_dfba = max(0.0, -v_glucose)
        ua_dfba = max(0.0, -v_acetate)
        uo_dfba = max(0.0, -v_oxygen)
        oxygen_effect = oxygen / (kinetics.Ko + oxygen + 1e-12)

        # ----------------------------------------------------
        # infer traits from realized dFBA
        # ----------------------------------------------------
        # glucose yield: infer mainly when acetate uptake is negligible
        if ug_dfba > cfg.uptake_eps and ua_dfba < cfg.phase_tol and oxygen_effect > cfg.uptake_eps:
            Yg_inst = mu / (ug_dfba * oxygen_effect + 1e-12)
            Yg_inst = max(0.0, Yg_inst)
            Yg_hat = (1.0 - cfg.ema) * Yg_hat + cfg.ema * Yg_inst

        # acetate yield: infer mainly when glucose uptake is negligible
        if ua_dfba > cfg.uptake_eps and ug_dfba < cfg.phase_tol and oxygen_effect > cfg.uptake_eps:
            Ya_inst = mu / (ua_dfba * oxygen_effect + 1e-12)
            Ya_inst = max(0.0, Ya_inst)
            Ya_hat = (1.0 - cfg.ema) * Ya_hat + cfg.ema * Ya_inst

        # alpha from acetate secretion per glucose uptake
        acetate_secretion = max(0.0, v_acetate)  # positive means secretion
        if ug_dfba > cfg.uptake_eps:
            alpha_inst = acetate_secretion / (ug_dfba + 1e-12)
            alpha_inst = max(0.0, alpha_inst)
            alpha_hat = (1.0 - cfg.ema) * alpha_hat + cfg.ema * alpha_inst

        row["mu_dfba"] = mu
        row["v_glucose"] = v_glucose
        row["v_acetate"] = v_acetate
        row["v_oxygen"] = v_oxygen
        row["ug_dfba"] = ug_dfba
        row["ua_dfba"] = ua_dfba
        row["uo_dfba"] = uo_dfba
        row["Yg_inferred"] = Yg_hat
        row["Ya_inferred"] = Ya_hat
        row["alpha_inferred"] = alpha_hat
        row["oxygen_effect"] = oxygen_effect
        rows.append(row)

        biomass_new = biomass * np.exp(mu * cfg.step_hours)

        glucose = max(
            0.0,
            update_concentration_eq7(glucose, v_glucose, biomass, mu, cfg.step_hours)
        )
        acetate = max(
            0.0,
            update_concentration_eq7(acetate, v_acetate, biomass, mu, cfg.step_hours)
        )
        oxygen = max(
            0.0,
            update_concentration_eq7(oxygen, v_oxygen, biomass, mu, cfg.step_hours)
        )
        biomass = biomass_new

    return pd.DataFrame(rows)


# ============================================================
# CRM using inferred traits from dFBA
# ============================================================
def simulate_crm_with_inferred_traits(
    df_traits: pd.DataFrame,
    kinetics: CRMKinetics,
    cfg: DFBAConfig,
) -> pd.DataFrame:
    t_grid = df_traits["time"].to_numpy(float)
    Yg_series = df_traits["Yg_inferred"].to_numpy(float)
    Ya_series = df_traits["Ya_inferred"].to_numpy(float)
    alpha_series = df_traits["alpha_inferred"].to_numpy(float)

    def trait_at(t: float):
        Yg_t = float(np.interp(t, t_grid, Yg_series))
        Ya_t = float(np.interp(t, t_grid, Ya_series))
        alpha_t = float(np.interp(t, t_grid, alpha_series))
        return Yg_t, Ya_t, alpha_t

    def rhs(t, y):
        X, G, A, O = y

        X = max(0.0, X)
        G = max(0.0, G)
        A = max(0.0, A)
        O = max(0.0, O)

        ug = michaelis_menten(G, kinetics.Vg, kinetics.Kg)
        ua = michaelis_menten(A, kinetics.Va, kinetics.Ka)
        uo = michaelis_menten(O, kinetics.Vo, kinetics.Ko)

        if kinetics.enable_acetate_gate and G > kinetics.glucose_gate_threshold:
            ua = 0.0

        oxygen_effect = O / (kinetics.Ko + O + 1e-12)
        Yg_t, Ya_t, alpha_t = trait_at(t)

        mu = Yg_t * ug * oxygen_effect + Ya_t * ua * oxygen_effect - kinetics.maintenance

        dX = X * mu
        dG = -X * ug
        dA = X * (alpha_t * ug - ua)
        dO = -X * uo
        return [dX, dG, dA, dO]

    y0 = [
        cfg.initial_biomass,
        cfg.initial_glucose,
        cfg.initial_acetate,
        cfg.initial_oxygen,
    ]

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, cfg.end_time_hours),
        y0=y0,
        t_eval=t_grid,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    out = pd.DataFrame({
        "time": sol.t,
        "biomass": np.maximum(0.0, sol.y[0]),
        "glucose": np.maximum(0.0, sol.y[1]),
        "acetate": np.maximum(0.0, sol.y[2]),
        "oxygen": np.maximum(0.0, sol.y[3]),
        "Yg_used": np.interp(sol.t, t_grid, Yg_series),
        "Ya_used": np.interp(sol.t, t_grid, Ya_series),
        "alpha_used": np.interp(sol.t, t_grid, alpha_series),
    })
    return out


# ============================================================
# Comparison / plotting
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
    df_dfba: pd.DataFrame,
    df_crm: pd.DataFrame,
    experiment_csv: Optional[str] = None,
    exp_time_col: str = "Time",
    exp_biomass_col: str = "Biomass",
):
    os.makedirs("hybrid_CRM_dfba_plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["biomass"], lw=2, label="dFBA")
    plt.plot(df_crm["time"], df_crm["biomass"], "--", lw=2, label="CRM from inferred yields")

    if experiment_csv is not None:
        exp_df = pd.read_csv(experiment_csv)
        plt.scatter(
            exp_df[exp_time_col],
            exp_df[exp_biomass_col],
            s=25,
            label="Experiment"
        )

    plt.xlabel("Time (h)")
    plt.ylabel("Biomass")
    plt.title("Biomass: dFBA vs CRM vs Experiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_CRM_dfba_plots/compare_biomass.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["glucose"], label="Glucose dFBA")
    plt.plot(df_crm["time"], df_crm["glucose"], "--", label="Glucose CRM")
    plt.plot(df_dfba["time"], df_dfba["acetate"], label="Acetate dFBA")
    plt.plot(df_crm["time"], df_crm["acetate"], "--", label="Acetate CRM")
    plt.plot(df_dfba["time"], df_dfba["oxygen"], label="Oxygen dFBA")
    plt.plot(df_crm["time"], df_crm["oxygen"], "--", label="Oxygen CRM")
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("Substrates: dFBA vs CRM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_CRM_dfba_plots/compare_substrates.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["ug_crm"], label="ug CRM")
    plt.plot(df_dfba["time"], df_dfba["ug_dfba"], "--", label="ug dFBA")
    plt.plot(df_dfba["time"], df_dfba["ua_crm"], label="ua CRM")
    plt.plot(df_dfba["time"], df_dfba["ua_dfba"], "--", label="ua dFBA")
    plt.plot(df_dfba["time"], df_dfba["uo_crm"], label="uo CRM")
    plt.plot(df_dfba["time"], df_dfba["uo_dfba"], "--", label="uo dFBA")
    plt.xlabel("Time (h)")
    plt.ylabel("Rate")
    plt.title("CRM MM uptake vs realized dFBA uptake")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_CRM_dfba_plots/compare_fluxes.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["Yg_inferred"], label="Yg inferred from dFBA")
    plt.plot(df_dfba["time"], df_dfba["Ya_inferred"], label="Ya inferred from dFBA")
    plt.plot(df_dfba["time"], df_dfba["alpha_inferred"], label="alpha inferred from dFBA")
    plt.xlabel("Time (h)")
    plt.ylabel("Inferred trait value")
    plt.title("Traits inferred from dFBA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_CRM_dfba_plots/inferred_traits.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["mu_dfba"], label="mu dFBA")
    plt.xlabel("Time (h)")
    plt.ylabel("Growth rate")
    plt.title("dFBA growth rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_CRM_dfba_plots/dfba_growth_rate.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    kinetics = CRMKinetics(
        Vg=11.5,
        Va=4.0,
        Vo=17.5,
        Kg=0.05,
        Ka=0.05,
        Ko=0.05,
        maintenance=0.0,
        enable_acetate_gate=True,
        glucose_gate_threshold=0.5,
        Yg_init=0.08,
        Ya_init=0.03,
        alpha_init=0.5,
    )

    cfg = DFBAConfig(
        model_name="textbook",
        step_hours=0.01,
        end_time_hours=10.0,
        initial_biomass=0.0045,
        initial_glucose=11.1,
        initial_acetate=0.0,
        initial_oxygen=50.0,
        experiment_csv="/Users/edwin/Downloads/plot-data (2).csv",
        plot=True
    )

    # 1) dFBA constrained by CRM MM uptakes
    df_dfba = run_crm_guided_mm_dfba_and_infer_traits(kinetics, cfg)

    # 2) CRM simulated using yields inferred from dFBA
    df_crm = simulate_crm_with_inferred_traits(df_dfba, kinetics, cfg)

    # save
    df_dfba.to_csv(cfg.output_dfba_csv, index=False)
    df_crm.to_csv(cfg.output_crm_csv, index=False)

    print(f"Saved: {cfg.output_dfba_csv}")
    print(f"Saved: {cfg.output_crm_csv}")

    if cfg.experiment_csv is not None:
        print("\ndFBA vs experiment")
        compare_to_experiment(
            df_dfba,
            cfg.experiment_csv,
            time_col=cfg.exp_time_col,
            biomass_col=cfg.exp_biomass_col,
        )

        print("CRM vs experiment")
        compare_to_experiment(
            df_crm,
            cfg.experiment_csv,
            time_col=cfg.exp_time_col,
            biomass_col=cfg.exp_biomass_col,
        )

    if cfg.plot:
        plot_results(
            df_dfba,
            df_crm,
            experiment_csv=cfg.experiment_csv,
            exp_time_col=cfg.exp_time_col,
            exp_biomass_col=cfg.exp_biomass_col,
        )


if __name__ == "__main__":
    main()