"""
crm_mm_dfba_yield_biomass_cap.py

CRM-guided dFBA with yield-based biomass cap.

What this script does
---------------------
1. Compute CRM Michaelis-Menten uptake rates at each time step:
      ug_crm, ua_crm, uo_crm

2. Use them as dFBA exchange bounds:
      EX_glucose lower bound = -ug_bound
      EX_acetate lower bound = -ua_bound
      EX_oxygen  lower bound = -uo_bound

3. Translate CRM yields into an FBA-understandable biomass cap:
      biomass_flux <= Yg * ug_bound * oxygen_effect
                    + Ya * ua_bound * oxygen_effect
                    - maintenance

4. Solve dFBA under:
      - exchange bounds
      - stoichiometric constraints
      - yield-based biomass cap

5. Update extracellular concentrations using the realized dFBA solution

6. Also simulate the CRM ODE directly and compare:
      CRM vs dFBA vs experiment

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
    V : float = 11.5
    Va: float = 4.0
    Vo: float = 13.5

    # Michaelis-Menten constants
    Kg: float = 0.05
    Ka: float = 0.05
    Ko: float = 0.05

    # Effective CRM yields / overflow / maintenance
    Yg: float = 0.08
    Ya: float = 0.03
    alpha: float = 0.5
    maintenance: float = 0.0

    # Optional acetate gate
    enable_acetate_gate: bool = True
    glucose_gate_threshold: float = 0.5


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
    use_biomass_cap: bool = True

    experiment_csv: Optional[str] = "/Users/edwin/Downloads/plot-data (2).csv"
    exp_time_col: str = "Time"
    exp_biomass_col: str = "Biomass"

    output_dfba_csv: str = "dfba_yield_cap_output.csv"
    output_crm_csv: str = "crm_yield_cap_output.csv"
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
# CRM uptake and biomass cap
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


def compute_biomass_cap(
    ug_bound: float,
    ua_bound: float,
    oxygen: float,
    kinetics: CRMKinetics,
) -> float:
    oxygen_effect = oxygen / (kinetics.Ko + oxygen + 1e-12)

    mu_cap = (
        kinetics.Yg * ug_bound * oxygen_effect
        + kinetics.Ya * ua_bound * oxygen_effect
        - kinetics.maintenance
    )

    return max(0.0, float(mu_cap))


# ============================================================
# CRM ODE
# ============================================================
def crm_rhs(t, y, kinetics: CRMKinetics):
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

    mu = (
        kinetics.Yg * ug * oxygen_effect
        + kinetics.Ya * ua * oxygen_effect
        - kinetics.maintenance
    )

    dX = X * mu
    dG = -X * ug
    dA = X * (kinetics.alpha * ug - ua)
    dO = -X * uo

    return [dX, dG, dA, dO]


def simulate_crm_only(kinetics: CRMKinetics, cfg: DFBAConfig) -> pd.DataFrame:
    t_eval = np.arange(0.0, cfg.end_time_hours + 1e-12, cfg.step_hours)

    y0 = [
        cfg.initial_biomass,
        cfg.initial_glucose,
        cfg.initial_acetate,
        cfg.initial_oxygen,
    ]

    sol = solve_ivp(
        fun=lambda t, y: crm_rhs(t, y, kinetics),
        t_span=(0.0, cfg.end_time_hours),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return pd.DataFrame({
        "time": sol.t,
        "biomass": np.maximum(0.0, sol.y[0]),
        "glucose": np.maximum(0.0, sol.y[1]),
        "acetate": np.maximum(0.0, sol.y[2]),
        "oxygen": np.maximum(0.0, sol.y[3]),
    })


# ============================================================
# dFBA with yield-based biomass cap
# ============================================================
def run_crm_guided_mm_dfba_with_biomass_cap(
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

        biomass_cap = compute_biomass_cap(
            ug_bound=ug_bound,
            ua_bound=ua_bound,
            oxygen=oxygen,
            kinetics=kinetics,
        )

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
            "biomass_cap": biomass_cap,
            "mu_dfba": np.nan,
            "v_glucose": np.nan,
            "v_acetate": np.nan,
            "v_oxygen": np.nan,
            "ug_dfba": np.nan,
            "ua_dfba": np.nan,
            "uo_dfba": np.nan,
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

            # CRM-guided uptake bounds
            m.reactions.get_by_id(cfg.ex_glucose).lower_bound = -abs(ug_bound)
            m.reactions.get_by_id(cfg.ex_acetate).lower_bound = -abs(ua_bound)
            m.reactions.get_by_id(cfg.ex_oxygen).lower_bound = -abs(uo_bound)

            # Yield-based biomass cap
            if cfg.use_biomass_cap:
                bio_rxn = m.reactions.get_by_id(cfg.biomass_reaction)
                bio_rxn.upper_bound = min(float(bio_rxn.upper_bound), biomass_cap)

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

        row["mu_dfba"] = mu
        row["v_glucose"] = v_glucose
        row["v_acetate"] = v_acetate
        row["v_oxygen"] = v_oxygen
        row["ug_dfba"] = ug_dfba
        row["ua_dfba"] = ua_dfba
        row["uo_dfba"] = uo_dfba
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
    os.makedirs("presentation_plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["biomass"], lw=2, label="dFBA with biomass cap")
    plt.plot(df_crm["time"], df_crm["biomass"], "--", lw=2, label="CRM ODE")

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
    plt.title("Biomass: CRM vs dFBA vs Experiment")
    plt.tight_layout()
    plt.savefig("constrained_dfba_plots/Biomass: CRM vs dFBA vs Experiment.png", dpi=300, bbox_inches="tight")
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
    plt.title("Substrates: CRM vs dFBA")
    plt.tight_layout()
    plt.savefig("constrained_dfba_plots/Substrates: CRM vs dFBA.png", dpi=300, bbox_inches="tight")
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
    plt.title("CRM uptake vs realized dFBA uptake")
    plt.tight_layout()
    plt.savefig("constrained_dfba_plots/CRM uptake vs realized dFBA uptake.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_dfba["time"], df_dfba["biomass_cap"], label="Yield-based biomass cap")
    plt.plot(df_dfba["time"], df_dfba["mu_dfba"], "--", label="Realized dFBA growth")
    plt.xlabel("Time (h)")
    plt.ylabel("Rate")
    plt.title("Biomass cap vs realized dFBA growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("constrained_dfba_plots/Extracellular_concentrations_CRM_dFBA.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    kinetics = CRMKinetics(
        Vg = 11.5,
        Va=4.0,
        Vo=17.5,
        Kg=0.05,
        Ka=0.05,
        Ko=0.05,
        Yg=0.08,
        Ya=0.03,
        alpha=0.5,
        maintenance=0.07,
        enable_acetate_gate=True,
        glucose_gate_threshold=0.5,
    )

    cfg = DFBAConfig(
        model_name="textbook",
        step_hours=0.01,
        end_time_hours=10.0,
        initial_biomass=0.0045,
        initial_glucose=11.1,
        initial_acetate=0.0,
        initial_oxygen=100.0,
        experiment_csv="/Users/edwin/Downloads/plot-data (2).csv",
        use_availability_caps=True,
        use_biomass_cap=True,
    )

    # CRM ODE
    df_crm = simulate_crm_only(kinetics, cfg)

    # dFBA with CRM uptake bounds + yield-based biomass cap
    df_dfba = run_crm_guided_mm_dfba_with_biomass_cap(kinetics, cfg)

    df_dfba.to_csv(cfg.output_dfba_csv, index=False)
    df_crm.to_csv(cfg.output_crm_csv, index=False)

    print(f"Saved: {cfg.output_dfba_csv}")
    print(f"Saved: {cfg.output_crm_csv}")

    if cfg.experiment_csv is not None:
        print("\nCRM vs experiment")
        compare_to_experiment(
            df_crm,
            cfg.experiment_csv,
            time_col=cfg.exp_time_col,
            biomass_col=cfg.exp_biomass_col,
        )

        print("dFBA vs experiment")
        compare_to_experiment(
            df_dfba,
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