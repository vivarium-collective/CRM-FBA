from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List

from cobra.io import read_sbml_model
from cobra import Model
from cobra.flux_analysis import pfba as cobra_pfba


# ============================================================
# CONFIG
# ============================================================
@dataclass
class PredictionConfig:
    predictions_csv: str = "predictions.csv"

    time_col: str = "time"
    biomass_col: str = "biomass"            # gDW/L
    galactose_col: str = "galactose"
    ethanol_col: str = "ethanol"
    mu_col: str = "mu"
    q_gal_col: str = "q_gal"
    q_eth_uptake_col: str = "q_eth_uptake"
    q_eth_secretion_col: str = "q_eth_secretion"
    gate_eth_col: str = "gate_eth"
    yield_col: str = "yield_ethanol_from_gal"

    # optional columns
    a_gal_col: str = "a_gal"
    a_eth_col: str = "a_eth"

    # optional scaling
    mu_scale: float = 1.0
    q_gal_scale: float = 1.0
    q_eth_scale: float = 1.0
    secretion_scale: float = 1.0

    default_ethanol_yield: float = 0.53


@dataclass
class DFBAConfig:
    model_path: str = "/Users/edwin/Downloads/iMM904.xml"

    biomass_rxn: str = "BIOMASS_SC5_notrace"
    ex_gal: str = "EX_gal_e"
    ex_eth: str = "EX_etoh_e"
    ex_o2: str = "EX_o2_e"
    ex_glc_remove: str = "EX_glc__D_e"

    step_hours: float = 0.1
    end_time_hours: Optional[float] = None

    initial_biomass: Optional[float] = None
    initial_galactose: Optional[float] = None
    initial_ethanol: Optional[float] = None
    initial_oxygen: float = 100.0

    oxygen_cap: float = 0.5

    use_pfba: bool = False
    use_availability_caps: bool = True

    # target tracking tolerances
    mu_tol_frac: float = 0.20
    uptake_tol_frac: float = 0.25
    secretion_tol_frac: float = 0.25

    # biomass cap around predicted growth
    use_biomass_cap: bool = True
    biomass_cap_scale: float = 1.25
    max_biomass_flux: float = 2.0

    # ethanol gate
    enable_ethanol_gate: bool = True
    gal_gate_threshold: float = 0.5

    # fallback / scoring
    max_exchange_flux: float = 100.0
    max_mu: float = 2.0

    # search around tracking bounds if strict solve fails
    search_scales: tuple = (0.5, 0.75, 1.0, 1.25)

    output_csv: str = "prediction_guided_adaptive_dfba_fixed.csv"
    plot_prefix: str = "prediction_guided_adaptive_fixed"
    debug_first_steps: int = 10


# ============================================================
# UTILITIES
# ============================================================
def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def availability_cap(conc: float, biomass: float, dt: float) -> float:
    if biomass <= 0.0 or dt <= 0.0:
        return 0.0
    return max(0.0, float(conc) / (float(biomass) * float(dt) + 1e-12))


def update_concentration(concentration: float, exchange_flux: float, biomass: float, mu: float, dt: float) -> float:
    """
    Uses COBRA exchange sign convention:
      uptake  -> negative exchange flux
      secretion -> positive exchange flux
    """
    uptake_like = -float(exchange_flux)

    if abs(mu) < 1e-12:
        return float(concentration) - uptake_like * float(biomass) * float(dt)

    return float(concentration) + (uptake_like * float(biomass) / float(mu)) * (1.0 - np.exp(float(mu) * float(dt)))


def validate_ids(model: Model, cfg: DFBAConfig):
    missing = []
    for rid in [cfg.biomass_rxn, cfg.ex_gal, cfg.ex_eth, cfg.ex_o2]:
        if rid not in model.reactions:
            missing.append(rid)
    if missing:
        raise ValueError(f"Missing reaction IDs in model: {missing}")


def open_base_medium(model: Model):
    for ex in model.exchanges:
        ex.lower_bound = 0.0
        ex.upper_bound = 1000.0

    for rid in [
        "EX_h_e", "EX_h2o_e", "EX_na1_e", "EX_k_e", "EX_pi_e",
        "EX_so4_e", "EX_nh4_e", "EX_cl_e", "EX_mg2_e", "EX_ca2_e"
    ]:
        if rid in model.reactions:
            rxn = model.reactions.get_by_id(rid)
            rxn.lower_bound = -1000.0
            rxn.upper_bound = 1000.0


def solve_model(model: Model, use_pfba: bool):
    return cobra_pfba(model) if use_pfba else model.optimize()


# ============================================================
# CONTROLLER
# ============================================================
class PredictionController:
    def __init__(self, pred_df: pd.DataFrame, cfg: PredictionConfig):
        self.df = pred_df.sort_values(cfg.time_col).reset_index(drop=True)
        self.cfg = cfg
        self.t = self.df[cfg.time_col].to_numpy(float)

        required = [
            cfg.time_col,
            cfg.biomass_col,
            cfg.galactose_col,
            cfg.ethanol_col,
            cfg.mu_col,
            cfg.q_gal_col,
            cfg.q_eth_uptake_col,
            cfg.q_eth_secretion_col,
            cfg.gate_eth_col,
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required prediction columns: {missing}")

    def interp(self, col: str, t: float, default: float = 0.0) -> float:
        if col not in self.df.columns:
            return float(default)
        return float(np.interp(t, self.t, self.df[col].to_numpy(float)))

    def targets(self, t: float) -> Dict[str, float]:
        mu = max(0.0, self.interp(self.cfg.mu_col, t)) * self.cfg.mu_scale
        qg = max(0.0, self.interp(self.cfg.q_gal_col, t)) * self.cfg.q_gal_scale
        qeu = max(0.0, self.interp(self.cfg.q_eth_uptake_col, t)) * self.cfg.q_eth_scale
        qes = max(0.0, self.interp(self.cfg.q_eth_secretion_col, t)) * self.cfg.secretion_scale
        gate = np.clip(self.interp(self.cfg.gate_eth_col, t, default=0.0), 0.0, 1.0)
        Y = self.interp(self.cfg.yield_col, t, default=self.cfg.default_ethanol_yield)

        return {
            "mu": mu,
            "q_gal": qg,
            "q_eth_uptake": qeu,
            "q_eth_secretion": qes,
            "gate_eth": gate,
            "yield_ethanol_from_gal": Y,
            "a_gal": self.interp(self.cfg.a_gal_col, t, default=np.nan),
            "a_eth": self.interp(self.cfg.a_eth_col, t, default=np.nan),
        }


# ============================================================
# CONSTRAINED STEP
# ============================================================
def apply_guided_bounds(
    m: Model,
    targets: Dict[str, float],
    biomass: float,
    gal: float,
    eth: float,
    o2: float,
    cfg: DFBAConfig,
    scale: float = 1.0,
):
    mu_pred = min(max(targets["mu"], 0.0), cfg.max_mu)
    qg_pred = min(max(targets["q_gal"], 0.0), cfg.max_exchange_flux)
    qeu_pred = min(max(targets["q_eth_uptake"], 0.0), cfg.max_exchange_flux)
    qes_pred = min(max(targets["q_eth_secretion"], 0.0), cfg.max_exchange_flux)
    Y = max(0.0, targets["yield_ethanol_from_gal"])

    if cfg.use_availability_caps:
        qg_pred = min(qg_pred, availability_cap(gal, biomass, cfg.step_hours))
        qeu_pred = min(qeu_pred, availability_cap(eth, biomass, cfg.step_hours))
        qo2_pred = min(cfg.oxygen_cap, availability_cap(o2, biomass, cfg.step_hours))
    else:
        qo2_pred = cfg.oxygen_cap

    qg_pred *= scale
    qeu_pred *= scale
    qes_pred *= scale

    gal_rxn = m.reactions.get_by_id(cfg.ex_gal)
    eth_rxn = m.reactions.get_by_id(cfg.ex_eth)
    o2_rxn = m.reactions.get_by_id(cfg.ex_o2)
    bio_rxn = m.reactions.get_by_id(cfg.biomass_rxn)

    # galactose uptake
    gal_tol = max(cfg.uptake_tol_frac * max(qg_pred, 1e-6), 1e-6)
    gal_lb = -(qg_pred + gal_tol)
    gal_ub = -(max(qg_pred - gal_tol, 0.0))
    gal_rxn.lower_bound = gal_lb
    gal_rxn.upper_bound = gal_ub

    # oxygen benchmark style
    o2_rxn.lower_bound = -abs(qo2_pred)
    o2_rxn.upper_bound = 1000.0

    # biomass cap
    mu_tol = max(cfg.mu_tol_frac * max(mu_pred, 1e-6), 1e-6)
    bio_rxn.lower_bound = max(0.0, mu_pred - mu_tol)
    bio_rxn.upper_bound = min(cfg.max_biomass_flux, mu_pred + mu_tol)
    if cfg.use_biomass_cap:
        mu_cap = min(cfg.max_biomass_flux, cfg.biomass_cap_scale * mu_pred)
        bio_rxn.upper_bound = max(1e-9, mu_cap)

    # ethanol handling
    # uptake after gate opens
    if cfg.enable_ethanol_gate and gal > cfg.gal_gate_threshold:
        eth_uptake_target = 0.0
    else:
        eth_uptake_target = qeu_pred * max(targets["gate_eth"], 0.0)

    # secretion ceiling from CRM yield
    # fixes the too-fermentative GSMM behavior
    max_eth_secretion_from_yield = Y * qg_pred
    max_eth_secretion = min(
        max(qes_pred, 0.0) + 0.05 * max(qes_pred, 1e-6),
        max_eth_secretion_from_yield + 0.05 * max(max_eth_secretion_from_yield, 1e-6)
    )

    eth_tol = max(cfg.uptake_tol_frac * max(eth_uptake_target, 1e-6), 1e-6)
    eth_lb = -(eth_uptake_target + eth_tol)
    eth_ub = max_eth_secretion

    eth_rxn.lower_bound = eth_lb
    eth_rxn.upper_bound = eth_ub

    return {
        "mu_pred": mu_pred,
        "qg_pred": qg_pred,
        "qeu_pred": qeu_pred,
        "qes_pred": qes_pred,
        "qo2_pred": qo2_pred,
        "Y": Y,
    }


def score_solution(sol, cfg: DFBAConfig, targets: Dict[str, float], guided_vals: Dict[str, float]) -> float:
    mu = max(0.0, float(sol.fluxes[cfg.biomass_rxn]))
    v_gal = float(sol.fluxes[cfg.ex_gal])
    v_eth = float(sol.fluxes[cfg.ex_eth])

    qg_dfba = max(0.0, -v_gal)
    qeu_dfba = max(0.0, -v_eth)
    qes_dfba = max(0.0, v_eth)

    score = 0.0
    score += 40.0 * (mu - guided_vals["mu_pred"]) ** 2
    score += 3.0 * (qg_dfba - guided_vals["qg_pred"]) ** 2
    score += 3.0 * (qeu_dfba - guided_vals["qeu_pred"]) ** 2
    score += 15.0 * (qes_dfba - min(guided_vals["qes_pred"], guided_vals["Y"] * guided_vals["qg_pred"])) ** 2
    return float(score)


def run_one_step(
    base_model: Model,
    biomass: float,
    gal: float,
    eth: float,
    o2: float,
    targets: Dict[str, float],
    cfg: DFBAConfig,
):
    best = None
    best_score = np.inf
    best_guided = None

    for scale in cfg.search_scales:
        with base_model as m:
            open_base_medium(m)
            m.objective = cfg.biomass_rxn

            guided_vals = apply_guided_bounds(
                m=m,
                targets=targets,
                biomass=biomass,
                gal=gal,
                eth=eth,
                o2=o2,
                cfg=cfg,
                scale=scale,
            )

            try:
                sol = solve_model(m, cfg.use_pfba)
            except Exception:
                sol = None

            if sol is None or sol.status != "optimal":
                continue

            s = score_solution(sol, cfg, targets, guided_vals)
            if s < best_score:
                best = sol
                best_score = s
                best_guided = guided_vals

    if best is None:
        # fallback
        mu = min(max(targets["mu"], 0.0), cfg.max_mu)
        qg = min(max(targets["q_gal"], 0.0), cfg.max_exchange_flux)
        qeu = min(max(targets["q_eth_uptake"], 0.0), cfg.max_exchange_flux)
        qes = min(max(targets["q_eth_secretion"], 0.0), cfg.max_exchange_flux)
        Y = max(0.0, targets["yield_ethanol_from_gal"])

        if cfg.use_availability_caps:
            qg = min(qg, availability_cap(gal, biomass, cfg.step_hours))
            qeu = min(qeu, availability_cap(eth, biomass, cfg.step_hours))
            qo2 = min(cfg.oxygen_cap, availability_cap(o2, biomass, cfg.step_hours))
        else:
            qo2 = cfg.oxygen_cap

        if cfg.enable_ethanol_gate and gal > cfg.gal_gate_threshold:
            qeu = 0.0

        qes = min(qes, Y * qg)

        return {
            "mode": "fallback",
            "mu": mu,
            "v_gal": -qg,
            "v_eth": qes - qeu,
            "v_o2": -qo2,
            "mu_pred": mu,
            "qg_pred": qg,
            "qeu_pred": qeu,
            "qes_pred": qes,
            "Y": Y,
        }

    mu = max(0.0, min(float(best.fluxes[cfg.biomass_rxn]), cfg.max_mu))
    v_gal = float(best.fluxes[cfg.ex_gal])
    v_eth = float(best.fluxes[cfg.ex_eth])
    v_o2 = float(best.fluxes[cfg.ex_o2])

    return {
        "mode": "guided",
        "mu": mu,
        "v_gal": v_gal,
        "v_eth": v_eth,
        "v_o2": v_o2,
        "mu_pred": best_guided["mu_pred"],
        "qg_pred": best_guided["qg_pred"],
        "qeu_pred": best_guided["qeu_pred"],
        "qes_pred": best_guided["qes_pred"],
        "Y": best_guided["Y"],
    }


# ============================================================
# MAIN SIMULATION
# ============================================================
def run_prediction_guided_dfba(pred_df: pd.DataFrame, p_cfg: PredictionConfig, d_cfg: DFBAConfig) -> pd.DataFrame:
    model = read_sbml_model(d_cfg.model_path)
    validate_ids(model, d_cfg)

    media = model.medium.copy()
    if d_cfg.ex_glc_remove in media:
        del media[d_cfg.ex_glc_remove]
    media[d_cfg.ex_gal] = 100.0
    media[d_cfg.ex_eth] = 0.0
    model.medium = media

    controller = PredictionController(pred_df, p_cfg)

    end_time = float(d_cfg.end_time_hours) if d_cfg.end_time_hours is not None else float(pred_df[p_cfg.time_col].iloc[-1])
    time_grid = np.arange(0.0, end_time + 1e-12, d_cfg.step_hours)

    biomass = float(d_cfg.initial_biomass if d_cfg.initial_biomass is not None else pred_df[p_cfg.biomass_col].iloc[0])
    gal = float(d_cfg.initial_galactose if d_cfg.initial_galactose is not None else pred_df[p_cfg.galactose_col].iloc[0])
    eth = float(d_cfg.initial_ethanol if d_cfg.initial_ethanol is not None else pred_df[p_cfg.ethanol_col].iloc[0])
    o2 = float(d_cfg.initial_oxygen)

    rows = []

    for i, t in enumerate(time_grid):
        targets = controller.targets(float(t))

        row = {
            "time": float(t),
            "biomass": biomass,
            "galactose": gal,
            "ethanol": eth,
            "oxygen": o2,
            "mu_pred": targets["mu"],
            "qg_pred_raw": targets["q_gal"],
            "qeu_pred_raw": targets["q_eth_uptake"],
            "qes_pred_raw": targets["q_eth_secretion"],
            "gate_eth": targets["gate_eth"],
            "a_gal": targets["a_gal"],
            "a_eth": targets["a_eth"],
            "yield_ethanol_from_gal": targets["yield_ethanol_from_gal"],
        }

        if t >= end_time:
            row.update({
                "tracking_mode": "terminal",
                "mu_dfba": np.nan,
                "v_galactose": np.nan,
                "v_ethanol": np.nan,
                "v_oxygen": np.nan,
                "qg_dfba": np.nan,
                "qeu_dfba": np.nan,
                "qes_dfba": np.nan,
                "uo_dfba": np.nan,
            })
            rows.append(row)
            break

        out = run_one_step(model, biomass, gal, eth, o2, targets, d_cfg)

        mu = out["mu"]
        v_gal = out["v_gal"]
        v_eth = out["v_eth"]
        v_o2 = out["v_o2"]

        row.update({
            "tracking_mode": out["mode"],
            "mu_dfba": mu,
            "v_galactose": v_gal,
            "v_ethanol": v_eth,
            "v_oxygen": v_o2,
            "qg_pred": out["qg_pred"],
            "qeu_pred": out["qeu_pred"],
            "qes_pred": out["qes_pred"],
            "qg_dfba": max(0.0, -v_gal),
            "qeu_dfba": max(0.0, -v_eth),
            "qes_dfba": max(0.0, v_eth),
            "uo_dfba": max(0.0, -v_o2),
        })
        rows.append(row)

        if i < d_cfg.debug_first_steps:
            print(
                f"t={t:.2f} | mode={out['mode']} | "
                f"mu_pred={out['mu_pred']:.4f}, mu_dfba={mu:.4f} | "
                f"qg_pred={out['qg_pred']:.4f}, qg_dfba={max(0.0, -v_gal):.4f} | "
                f"qeu_pred={out['qeu_pred']:.4f}, qeu_dfba={max(0.0, -v_eth):.4f} | "
                f"qes_pred={out['qes_pred']:.4f}, qes_dfba={max(0.0, v_eth):.4f}"
            )

        biomass = biomass * np.exp(mu * d_cfg.step_hours)
        gal = max(0.0, update_concentration(gal, v_gal, row["biomass"], mu, d_cfg.step_hours))
        eth = max(0.0, update_concentration(eth, v_eth, row["biomass"], mu, d_cfg.step_hours))
        o2 = max(0.0, update_concentration(o2, v_o2, row["biomass"], mu, d_cfg.step_hours))

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================
def plot_results(
    pred_df: pd.DataFrame,
    dfba_df: pd.DataFrame,
    p_cfg: PredictionConfig,
    d_cfg: DFBAConfig,
    gDW_per_cell: float = 4.765e-12,
    concentration_scale: float = 1000.0,
):
    t_pred = pred_df[p_cfg.time_col].to_numpy(float)

    # --------------------------------------------------------
    # Convert biomass to experimental units: cells/mL
    # biomass[gDW/L] -> cells/mL = biomass / (1000 * gDW_per_cell)
    # --------------------------------------------------------
    pred_cells_ml = pred_df["cells_per_ml"].to_numpy(float)
    dfba_cells_ml = dfba_df["biomass"].to_numpy(float) / (1000.0 * gDW_per_cell)

    plt.figure(figsize=(10, 6))
    plt.plot(t_pred, pred_cells_ml, label="Prediction", lw=2.5)
    plt.plot(dfba_df["time"], dfba_cells_ml, label="Prediction-guided dFBA", lw=2.5)
    plt.xlabel("Time (h)")
    plt.ylabel("Population (cells/mL)")
    plt.title("Prediction vs guided dFBA biomass")
    plt.legend()
    save_plot(f"{d_cfg.plot_prefix}_biomass_cells_ml.png")

    # --------------------------------------------------------
    # Convert substrates back to the adaptive/CRM units
    # pred_df stores:
    #   galactose_raw, ethanol_raw  -> original adaptive units
    #   galactose, ethanol          -> scaled units for dFBA guidance
    #
    # So compare:
    #   prediction raw
    #   dFBA scaled / concentration_scale
    # --------------------------------------------------------
    pred_gal_raw = pred_df["galactose_raw"].to_numpy(float)
    pred_eth_raw = pred_df["ethanol_raw"].to_numpy(float)

    dfba_gal_raw = dfba_df["galactose"].to_numpy(float) / concentration_scale
    dfba_eth_raw = dfba_df["ethanol"].to_numpy(float) / concentration_scale

    plt.figure(figsize=(10, 6))
    plt.plot(t_pred, pred_gal_raw, label="Pred galactose", lw=2)
    plt.plot(t_pred, pred_eth_raw, label="Pred ethanol", lw=2)
    plt.plot(dfba_df["time"], dfba_gal_raw, "--", label="dFBA galactose", lw=2)
    plt.plot(dfba_df["time"], dfba_eth_raw, "--", label="dFBA ethanol", lw=2)
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (adaptive units)")
    plt.title("Extracellular states")
    plt.legend()
    save_plot(f"{d_cfg.plot_prefix}_states_raw_units.png")

    # --------------------------------------------------------
    # Flux plots stay the same
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(dfba_df["time"], dfba_df["qg_pred"], label="qg pred", lw=2)
    plt.plot(dfba_df["time"], dfba_df["qg_dfba"], "--", label="qg dFBA", lw=2)
    plt.plot(dfba_df["time"], dfba_df["qeu_pred"], label="qeth uptake pred", lw=2)
    plt.plot(dfba_df["time"], dfba_df["qeu_dfba"], "--", label="qeth uptake dFBA", lw=2)
    plt.plot(dfba_df["time"], dfba_df["qes_pred"], label="qeth secretion pred", lw=2)
    plt.plot(dfba_df["time"], dfba_df["qes_dfba"], "--", label="qeth secretion dFBA", lw=2)
    plt.xlabel("Time (h)")
    plt.ylabel("Rate")
    plt.title("Prediction-guided uptake/secretion vs realized dFBA")
    plt.legend()
    save_plot(f"{d_cfg.plot_prefix}_fluxes.png")

    plt.figure(figsize=(10, 6))
    plt.plot(dfba_df["time"], dfba_df["mu_pred"], label="mu pred", lw=2)
    plt.plot(dfba_df["time"], dfba_df["mu_dfba"], "--", label="mu dFBA", lw=2)
    plt.xlabel("Time (h)")
    plt.ylabel("Growth rate")
    plt.title("Predicted vs realized growth")
    plt.legend()
    save_plot(f"{d_cfg.plot_prefix}_growth.png")


# ============================================================
# MAIN
# ============================================================
def main():
    p_cfg = PredictionConfig(
        predictions_csv="predictions.csv",
        default_ethanol_yield=0.53,
        mu_scale=1.0,
        q_gal_scale=1.0,
        q_eth_scale=1.0,
        secretion_scale=1.0,
    )

    pred_df = pd.read_csv(p_cfg.predictions_csv)

    d_cfg = DFBAConfig(
        model_path="/Users/edwin/Downloads/iMM904.xml",
        step_hours=0.1,
        end_time_hours=float(pred_df[p_cfg.time_col].iloc[-1]),
        initial_biomass=float(pred_df[p_cfg.biomass_col].iloc[0]),
        initial_galactose=float(pred_df[p_cfg.galactose_col].iloc[0]),
        initial_ethanol=float(pred_df[p_cfg.ethanol_col].iloc[0]),
        initial_oxygen=100.0,
        oxygen_cap=0.5,
        use_pfba=False,
        use_availability_caps=True,
        mu_tol_frac=0.1,
        uptake_tol_frac=0.1,
        secretion_tol_frac=0.1,
        use_biomass_cap=True,
        biomass_cap_scale=1.25,
        max_biomass_flux=2.0,
        enable_ethanol_gate=True,
        gal_gate_threshold=0.5,
        max_exchange_flux=100.0,
        max_mu=2.0,
        search_scales=(0.5, 0.75, 1.0, 1.25),
        output_csv="prediction_guided_adaptive_dfba_fixed.csv",
        plot_prefix="prediction_guided_adaptive_fixed",
        debug_first_steps=10,
    )

    dfba_df = run_prediction_guided_dfba(pred_df, p_cfg, d_cfg)
    dfba_df.to_csv(d_cfg.output_csv, index=False)
    print(f"Saved {d_cfg.output_csv}")

    plot_results(
        pred_df,
        dfba_df,
        p_cfg,
        d_cfg,
        gDW_per_cell=4.765e-12,
        concentration_scale=1000.0,
    )


if __name__ == "__main__":
    main()