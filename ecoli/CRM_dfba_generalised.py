"""
crm_dfba_benchmark_general_same_math.py

Generalized:
- Uses the same dFBA logic (availability cap Eq(5), biomass update Eq(6), conc update Eq(7))
- Uses the same calibration logic:
    - yield = mu / uptake  (single-point at vmax caps)
    - acetate_per_glucose = acetate_secretion / glucose_uptake (single-point)
- Uses the same CRM ODE as your original:
    dX/dt = X * (Y_g*u_g + Y_a*u_a - maintenance)
    dG/dt = -X * u_g
    dA/dt =  X * (acetate_per_glucose*u_g - u_a)
  with the optional glucose gate that sets acetate uptake to 0 until glucose < threshold.

Generalization approach:
- Put model-specific IDs and numbers into a config dataclass
- Still keeps the "glucose/oxygen/acetate" semantics because your math is specific to that story
- If you later want different carbon/byproduct pairs, you only change the config and keys.

Deps: cobra, numpy, scipy, matplotlib, pandas
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import cobra
from cobra.io import load_model
from cobra.flux_analysis import pfba as cobra_pfba


# ============================================================
# Data containers (kept consistent with your original)
# ============================================================
@dataclass
class CRMParameters:
    Vmax_glucose: float
    Vmax_acetate: float
    Y_glucose: float
    Y_acetate: float
    Km_glucose: float
    Km_acetate: float
    maintenance: float
    acetate_per_glucose: float  # mmol acetate produced per mmol glucose consumed


@dataclass
class BenchmarkResults:
    # Experimental (biomass only)
    time_experimental: np.ndarray
    biomass_experimental: np.ndarray

    # dFBA
    time_dfba: np.ndarray
    biomass_dfba: np.ndarray
    glucose_dfba: np.ndarray
    acetate_dfba: np.ndarray
    oxygen_dfba: np.ndarray
    growth_rate_dfba: np.ndarray  # mu(t)

    # CRM
    time_crm: np.ndarray
    biomass_crm: np.ndarray
    glucose_crm: np.ndarray
    acetate_crm: np.ndarray

    # Errors (dimensionless NMSE, biomass only)
    biomass_nmse_experiment_vs_dfba: float
    biomass_nmse_experiment_vs_crm: float
    biomass_nmse_dfba_vs_crm: float

    # Calibrations
    glucose_yield: float
    acetate_yield: float
    acetate_per_glucose: float


# ============================================================
# Generalizable config (but uses your exact defaults/values)
# ============================================================
@dataclass
class BenchmarkConfig:
    # Experimental CSV
    experimental_csv_path: str = "/Users/edwin/Downloads/plot-data (2).csv"
    time_column: str = "Time"
    biomass_column: str = "Biomass"
    biomass_scale: float = 0.8

    # COBRA model
    model_name: str = "textbook"

    # Reaction IDs (exactly as your script)
    ex_glucose: str = "EX_glc__D_e"
    ex_oxygen: str = "EX_o2_e"
    ex_acetate: str = "EX_ac_e"
    biomass_reaction: str = "Biomass_Ecoli_core"

    # Byproducts to allow secretion (paper-style) (exact list)
    byproduct_exchanges: List[str] = None

    # Minimal inorganic base medium (exact list)
    base_medium_exchanges: List[str] = None

    # Simulation horizon (exact defaults)
    step_hours: float = 0.01
    end_time_hours: float = 10.0

    # Initial conditions (exact defaults)
    initial_biomass: float = 0.0045  # gDW/L
    initial_concentrations: Dict[str, float] = None  # keys: glucose, oxygen, acetate (mmol/L)

    # Paper-ish caps (exact defaults)
    vmax_glucose: float = 11.5  # mmol/gDW/hr
    vmax_oxygen: float = 13.8   # mmol/gDW/hr

    # CRM kinetics (exact defaults)
    Km_glucose: float = 0.05  # mmol/L
    Km_acetate: float = 0.05  # mmol/L

    # CRM maintenance (exact defaults)
    crm_maintenance: float = 0.0


    # Optional acetate gate (exact defaults)
    enable_acetate_gate: bool = True
    glucose_gate_threshold: float = 0.5  # mmol/L

    # Solver option (exact default)
    use_pfba: bool = False

    def __post_init__(self):
        if self.byproduct_exchanges is None:
            self.byproduct_exchanges = ["EX_ac_e", "EX_etoh_e", "EX_for_e", "EX_lac__D_e", "EX_succ_e"]
        if self.base_medium_exchanges is None:
            self.base_medium_exchanges = [
                "EX_h_e", "EX_h2o_e", "EX_na1_e", "EX_k_e", "EX_pi_e",
                "EX_so4_e", "EX_nh4_e", "EX_cl_e", "EX_mg2_e", "EX_ca2_e",
            ]
        if self.initial_concentrations is None:
            self.initial_concentrations = dict(glucose=11.1, oxygen=50, acetate=0.0)


# ============================================================
# Shared helpers (EXACT same logic as your original)
# ============================================================
def solve_fba(model: cobra.Model, use_pfba: bool):
    return cobra_pfba(model) if (use_pfba and cobra_pfba is not None) else model.optimize()


def open_inorganic_base_medium(
    model: cobra.Model,
    *,
    base_medium_exchanges,
    ex_oxygen: str,
    oxygen_cap: Optional[float] = None,
) -> None:
    # close all uptake; allow secretion
    for ex in model.exchanges:
        ex.lower_bound = 0.0
        ex.upper_bound = 1000.0

    # open inorganic base
    for rid in base_medium_exchanges:
        if rid in model.reactions:
            r = model.reactions.get_by_id(rid)
            r.lower_bound = -1000.0
            r.upper_bound = 1000.0

    # optional oxygen cap
    if oxygen_cap is not None and ex_oxygen in model.reactions:
        r = model.reactions.get_by_id(ex_oxygen)
        r.lower_bound = -abs(float(oxygen_cap))
        r.upper_bound = 1000.0


def availability_cap_mmol_per_gdw_hr(concentration_mmol_per_L: float, biomass_gdw_per_L: float, step_hr: float) -> float:
    # Eq (5): cap = C / (X * dt)
    if biomass_gdw_per_L <= 0 or step_hr <= 0:
        return 0.0
    return max(0.0, float(concentration_mmol_per_L) / (float(biomass_gdw_per_L) * float(step_hr)))


def update_concentration_eq7(
    concentration: float,
    exchange_flux_mmol_per_gdw_hr: float,
    biomass_gdw_per_L: float,
    growth_rate_per_hr: float,
    step_hr: float
) -> float:
    # Eq (7): analytic update; uptake is negative in COBRA => substrate_uptake = -v_ex
    substrate_uptake = -float(exchange_flux_mmol_per_gdw_hr)
    if abs(growth_rate_per_hr) < 1e-12:
        return float(concentration) - substrate_uptake * float(biomass_gdw_per_L) * float(step_hr)

    return float(concentration) + (substrate_uptake * float(biomass_gdw_per_L) / float(growth_rate_per_hr)) * (
        1.0 - np.exp(float(growth_rate_per_hr) * float(step_hr))
    )


def normalized_mse(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return float(np.mean(((np.asarray(a) - np.asarray(b)) / float(scale)) ** 2))


def validate_ids(model: cobra.Model, cfg: BenchmarkConfig) -> None:
    missing = []
    for rid in [cfg.ex_glucose, cfg.ex_oxygen, cfg.ex_acetate, cfg.biomass_reaction]:
        if rid not in model.reactions:
            missing.append(rid)
    if missing:
        raise ValueError(f"Missing reaction IDs in model: {missing}")


# ============================================================
# dFBA (EXACT same behavior, just parameterized via cfg)
# ============================================================
def run_dfba_batch(model: cobra.Model, cfg: BenchmarkConfig) -> Dict[str, np.ndarray]:
    time_grid = np.arange(0.0, cfg.end_time_hours + 1e-12, cfg.step_hours)

    biomass = float(cfg.initial_biomass)
    concentrations = dict(cfg.initial_concentrations)  # keys: glucose, oxygen, acetate

    outputs = {k: [] for k in ["time", "biomass", "glucose", "oxygen", "acetate", "growth_rate"]}

    for t in time_grid:
        outputs["time"].append(t)
        outputs["biomass"].append(biomass)
        outputs["glucose"].append(concentrations["glucose"])
        outputs["oxygen"].append(concentrations["oxygen"])
        outputs["acetate"].append(concentrations["acetate"])

        if t >= cfg.end_time_hours:
            break

        vmax_glucose_eff = min(cfg.vmax_glucose, availability_cap_mmol_per_gdw_hr(concentrations["glucose"], biomass, cfg.step_hours))
        vmax_oxygen_eff  = min(cfg.vmax_oxygen,  availability_cap_mmol_per_gdw_hr(concentrations["oxygen"],  biomass, cfg.step_hours))

        with model as m:
            m.objective = cfg.biomass_reaction
            open_inorganic_base_medium(
                m,
                base_medium_exchanges=cfg.base_medium_exchanges,
                ex_oxygen=cfg.ex_oxygen,
                oxygen_cap=None,
            )

            # allow secretion of typical byproducts
            for rid in cfg.byproduct_exchanges:
                if rid in m.reactions:
                    r = m.reactions.get_by_id(rid)
                    r.upper_bound = 1000.0
                    r.lower_bound = 0.0

            m.reactions.get_by_id(cfg.ex_glucose).lower_bound = -vmax_glucose_eff
            m.reactions.get_by_id(cfg.ex_oxygen).lower_bound  = -vmax_oxygen_eff

            # allow acetate uptake if acetate present (EXACT as your script)
            if concentrations["acetate"] > 1e-9 and cfg.ex_acetate in m.reactions:
                vmax_acetate_eff = availability_cap_mmol_per_gdw_hr(concentrations["acetate"], biomass, cfg.step_hours)
                m.reactions.get_by_id(cfg.ex_acetate).lower_bound = -vmax_acetate_eff

            solution = solve_fba(m, cfg.use_pfba)
            if solution.status != "optimal":
                break

            growth_rate = float(solution.objective_value)
            v_glucose = float(solution.fluxes[cfg.ex_glucose])
            v_oxygen  = float(solution.fluxes[cfg.ex_oxygen])
            v_acetate = float(solution.fluxes[cfg.ex_acetate])

        outputs["growth_rate"].append(growth_rate)

        # Eq (6): biomass update
        biomass_new = biomass * float(np.exp(growth_rate * cfg.step_hours))

        # Eq (7): extracellular updates
        concentrations["glucose"] = max(0.0, update_concentration_eq7(concentrations["glucose"], v_glucose, biomass, growth_rate, cfg.step_hours))
        concentrations["oxygen"]  = max(0.0, update_concentration_eq7(concentrations["oxygen"],  v_oxygen,  biomass, growth_rate, cfg.step_hours))
        concentrations["acetate"] = max(0.0, update_concentration_eq7(concentrations["acetate"], v_acetate, biomass, growth_rate, cfg.step_hours))

        biomass = biomass_new

    # Pad growth_rate to align lengths (last step has no mu)
    if len(outputs["growth_rate"]) < len(outputs["time"]):
        outputs["growth_rate"].append(outputs["growth_rate"][-1] if outputs["growth_rate"] else 0.0)

    return {k: np.array(v, dtype=float) for k, v in outputs.items()}


# ============================================================
# Calibrations (EXACT same logic)
# ============================================================
def calibrate_growth_yield(
    model: cobra.Model,
    *,
    carbon_exchange: str,
    vmax_carbon: float,
    vmax_oxygen: float,
    ex_oxygen: str,
    biomass_reaction: str,
    byproduct_exchanges,
    base_medium_exchanges,
    use_pfba: bool = False,
) -> Tuple[float, float, float]:
    """
    Returns (growth_rate, carbon_uptake, yield = growth_rate / carbon_uptake)
    """
    with model as m:
        m.objective = biomass_reaction
        open_inorganic_base_medium(m, base_medium_exchanges=base_medium_exchanges, ex_oxygen=ex_oxygen)

        for rid in byproduct_exchanges:
            if rid in m.reactions:
                m.reactions.get_by_id(rid).upper_bound = 1000.0

        m.reactions.get_by_id(carbon_exchange).lower_bound = -abs(float(vmax_carbon))
        m.reactions.get_by_id(ex_oxygen).lower_bound = -abs(float(vmax_oxygen))

        sol = solve_fba(m, use_pfba)
        if sol.status != "optimal":
            return 0.0, 0.0, 0.0

        growth_rate = float(sol.objective_value)
        carbon_uptake = max(0.0, -float(sol.fluxes[carbon_exchange]))
        yield_coeff = (growth_rate / carbon_uptake) if carbon_uptake > 1e-12 else 0.0
        return growth_rate, carbon_uptake, yield_coeff


def calibrate_acetate_overflow(
    model: cobra.Model,
    *,
    vmax_glucose: float,
    vmax_oxygen: float,
    ex_glucose: str,
    ex_oxygen: str,
    ex_acetate: str,
    biomass_reaction: str,
    byproduct_exchanges,
    base_medium_exchanges,
    use_pfba: bool = False,
) -> float:
    """
    acetate_per_glucose = acetate_secretion / glucose_uptake
    """
    with model as m:
        m.objective = biomass_reaction
        open_inorganic_base_medium(m, base_medium_exchanges=base_medium_exchanges, ex_oxygen=ex_oxygen)

        for rid in byproduct_exchanges:
            if rid in m.reactions:
                m.reactions.get_by_id(rid).upper_bound = 1000.0

        m.reactions.get_by_id(ex_glucose).lower_bound = -abs(float(vmax_glucose))
        m.reactions.get_by_id(ex_oxygen).lower_bound  = -abs(float(vmax_oxygen))

        sol = solve_fba(m, use_pfba)
        if sol.status != "optimal":
            return 0.0

        glucose_uptake = max(1e-12, -float(sol.fluxes[ex_glucose]))
        acetate_secretion = max(0.0, float(sol.fluxes[ex_acetate]))
        return acetate_secretion / glucose_uptake


# ============================================================
# CRM simulation (EXACT same ODEs/logic, now parameterized)
# ============================================================
def simulate_crm_batch(
    params: CRMParameters,
    *,
    initial_biomass: float,
    initial_glucose: float,
    initial_acetate: float,
    step_hours: float,
    end_time_hours: float,
    enable_acetate_gate: bool,
    glucose_gate_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_grid = np.arange(0.0, end_time_hours + 1e-12, step_hours)

    def rhs(_, state):
        biomass, glucose, acetate = state

        glucose_uptake_fraction = glucose / (params.Km_glucose + glucose + 1e-12)
        acetate_uptake_fraction = acetate / (params.Km_acetate + acetate + 1e-12)

        glucose_uptake_rate = params.Vmax_glucose * glucose_uptake_fraction
        acetate_uptake_rate = params.Vmax_acetate * acetate_uptake_fraction

        if enable_acetate_gate and glucose > glucose_gate_threshold:
            acetate_uptake_rate = 0.0

        growth_rate = params.Y_glucose * glucose_uptake_rate + params.Y_acetate * acetate_uptake_rate

        d_biomass = biomass * (growth_rate - params.maintenance)
        d_glucose = -biomass * glucose_uptake_rate
        d_acetate = biomass * (params.acetate_per_glucose * glucose_uptake_rate - acetate_uptake_rate)

        return [d_biomass, d_glucose, d_acetate]

    sol = solve_ivp(
        rhs,
        (0.0, float(end_time_hours)),
        [float(initial_biomass), float(initial_glucose), float(initial_acetate)],
        t_eval=time_grid,
        rtol=1e-12,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    biomass = sol.y[0, :]
    glucose = sol.y[1, :]
    acetate = sol.y[2, :]
    return time_grid, biomass, glucose, acetate


# ============================================================
# Plotting (same plots as your original)
# ============================================================
def plot_benchmark(results: BenchmarkResults) -> None:
    plt.figure()
    plt.plot(results.time_experimental, results.biomass_experimental, lw=2.5, color="red", label="Experiment")
    plt.plot(results.time_dfba, results.biomass_dfba, lw=2, label="dFBA")
    plt.plot(results.time_crm, results.biomass_crm, lw=2, color="green", label="CRM")

    plt.text(
        0.03, 0.95,
        f"NMSE(Exp,dFBA)={results.biomass_nmse_experiment_vs_dfba:.3f}\n"
        f"NMSE(Exp,CRM) ={results.biomass_nmse_experiment_vs_crm:.3f}\n"
        f"NMSE(dFBA,CRM)={results.biomass_nmse_dfba_vs_crm:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.5")
    )
    plt.xlabel("Time (h)")
    plt.ylabel("Biomass (gDW/L)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("biomass", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(results.time_dfba, results.glucose_dfba, label="dFBA glucose")
    plt.plot(results.time_dfba, results.acetate_dfba, label="dFBA acetate")
    plt.plot(results.time_crm, results.glucose_crm, "--", label="CRM glucose")
    plt.plot(results.time_crm, results.acetate_crm, "--", label="CRM acetate")
    plt.xlabel("Time (hr)")
    plt.ylabel("Concentration (mmol/L)")
    plt.title("Extracellular concentrations: CRM vs dFBA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Extracellular_concentrations_CRM_dFBA.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Entrypoint (same workflow, but config-driven)
# ============================================================
def run_benchmark(cfg: Optional[BenchmarkConfig] = None, plot: bool = True, **overrides) -> BenchmarkResults:
    """
    Same pipeline as your script, but "generalizable" via cfg + overrides.

    Examples:
      run_benchmark()  # uses exact defaults from your script
      run_benchmark(experimental_csv_path="...", step_hours=0.1)
      cfg = BenchmarkConfig(model_name="iJO1366", ex_glucose="EX_glc__D_e", ...)
      run_benchmark(cfg)
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    # allow quick overrides (same pattern you used)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise AttributeError(f"Unknown config field: {k}")

    experimental_df = pd.read_csv(cfg.experimental_csv_path)
    time_experimental = experimental_df[cfg.time_column].to_numpy(dtype=float)
    biomass_experimental = experimental_df[cfg.biomass_column].to_numpy(dtype=float)

    model = load_model(cfg.model_name)
    validate_ids(model, cfg)

    dfba = run_dfba_batch(model, cfg)

    _, _, glucose_yield = calibrate_growth_yield(
        model,
        carbon_exchange=cfg.ex_glucose,
        vmax_carbon=cfg.vmax_glucose,
        vmax_oxygen=cfg.vmax_oxygen,
        ex_oxygen=cfg.ex_oxygen,
        biomass_reaction=cfg.biomass_reaction,
        byproduct_exchanges=cfg.byproduct_exchanges,
        base_medium_exchanges=cfg.base_medium_exchanges,
        use_pfba=cfg.use_pfba,
    )

    _, _, acetate_yield = calibrate_growth_yield(
        model,
        carbon_exchange=cfg.ex_acetate,
        vmax_carbon=10.0,               # EXACT as your script
        vmax_oxygen=cfg.vmax_oxygen,
        ex_oxygen=cfg.ex_oxygen,
        biomass_reaction=cfg.biomass_reaction,
        byproduct_exchanges=cfg.byproduct_exchanges,
        base_medium_exchanges=cfg.base_medium_exchanges,
        use_pfba=cfg.use_pfba,
    )

    acetate_per_glucose = calibrate_acetate_overflow(
        model,
        vmax_glucose=cfg.vmax_glucose,
        vmax_oxygen=cfg.vmax_oxygen,
        ex_glucose=cfg.ex_glucose,
        ex_oxygen=cfg.ex_oxygen,
        ex_acetate=cfg.ex_acetate,
        biomass_reaction=cfg.biomass_reaction,
        byproduct_exchanges=cfg.byproduct_exchanges,
        base_medium_exchanges=cfg.base_medium_exchanges,
        use_pfba=cfg.use_pfba,
    )

    crm_params = CRMParameters(
        Vmax_glucose=cfg.vmax_glucose,
        Vmax_acetate=10.0,              # EXACT as your script
        Y_glucose=glucose_yield,
        Y_acetate=acetate_yield,
        Km_glucose=cfg.Km_glucose,
        Km_acetate=cfg.Km_acetate,
        maintenance=cfg.crm_maintenance,
        acetate_per_glucose=acetate_per_glucose,
    )

    time_crm, biomass_crm, glucose_crm, acetate_crm = simulate_crm_batch(
        crm_params,
        initial_biomass=cfg.initial_biomass,
        initial_glucose=cfg.initial_concentrations["glucose"],
        initial_acetate=cfg.initial_concentrations["acetate"],
        step_hours=cfg.step_hours,
        end_time_hours=cfg.end_time_hours,
        enable_acetate_gate=cfg.enable_acetate_gate,
        glucose_gate_threshold=cfg.glucose_gate_threshold,
    )

    # Errors (biomass only)
    biomass_dfba_on_exp = np.interp(time_experimental, dfba["time"], dfba["biomass"])
    biomass_crm_on_exp  = np.interp(time_experimental, time_crm, biomass_crm)

    nmse_exp_dfba = normalized_mse(biomass_dfba_on_exp, biomass_experimental, cfg.biomass_scale)
    nmse_exp_crm  = normalized_mse(biomass_crm_on_exp,  biomass_experimental, cfg.biomass_scale)
    nmse_dfba_crm  = normalized_mse(biomass_dfba_on_exp, biomass_crm_on_exp,  cfg.biomass_scale)

    results = BenchmarkResults(
        time_experimental=time_experimental,
        biomass_experimental=biomass_experimental,

        time_dfba=dfba["time"],
        biomass_dfba=dfba["biomass"],
        glucose_dfba=dfba["glucose"],
        acetate_dfba=dfba["acetate"],
        oxygen_dfba=dfba["oxygen"],
        growth_rate_dfba=dfba["growth_rate"],

        time_crm=time_crm,
        biomass_crm=biomass_crm,
        glucose_crm=glucose_crm,
        acetate_crm=acetate_crm,

        biomass_nmse_experiment_vs_dfba=nmse_exp_dfba,
        biomass_nmse_experiment_vs_crm=nmse_exp_crm,
        biomass_nmse_dfba_vs_crm=nmse_dfba_crm,

        glucose_yield=glucose_yield,
        acetate_yield=acetate_yield,
        acetate_per_glucose=acetate_per_glucose,
    )

    if plot:
        plot_benchmark(results)

    return results


if __name__ == "__main__":
    run_benchmark()