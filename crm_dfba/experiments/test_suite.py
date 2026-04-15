"""
CRM-FBA experiment suite
========================

Runs a registry of diverse CRM-coupled dFBA experiments, saves a result
plot per experiment, and generates an HTML report at ``doc/index.html``
in the style of the viva-munk report. Run:

    python -m crm_dfba.experiments.test_suite           # runs all, opens browser
    python -m crm_dfba.experiments.test_suite --no-open
    python -m crm_dfba.experiments.test_suite --only diauxie overflow
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import os
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from process_bigraph import allocate_core

from crm_dfba import CRMDynamicFBA
from crm_dfba.models import get_model_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = REPO_ROOT / "doc"
_core = allocate_core()


# ---------------------------------------------------------------------------
# Experiment container
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    name: str
    title: str
    phenomenon: str
    description: str
    build_cfg: Callable[[], Dict[str, Any]]
    initial_substrates: Dict[str, float]
    initial_biomass: float = 0.01
    steps: int = 240
    dt: float = 0.05
    series_override: Optional[Callable[["Experiment"], Dict[str, Any]]] = None
    runner_override: Optional[Callable[["Experiment"], "RunResult"]] = None
    extra_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    experiment: Experiment
    t: np.ndarray
    biomass: np.ndarray
    substrates: Dict[str, np.ndarray]
    wall_clock_s: float
    # CRM components (recorded every step)
    crm_uptakes: Dict[str, np.ndarray] = field(default_factory=dict)
    # FBA results: realized exchange fluxes (negative = uptake) and growth rate
    fba_fluxes: Dict[str, np.ndarray] = field(default_factory=dict)
    mu: Optional[np.ndarray] = None
    # CRM-internal state (e.g. adaptive allocation A)
    crm_internal: Dict[str, np.ndarray] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    # Community results (set only by the multi-consumer runner)
    species: Optional[Dict[str, Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------
def _build_process(cfg):
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return CRMDynamicFBA(config=clean, core=_core)


def _simulate(exp: Experiment) -> RunResult:
    proc = _build_process(exp.build_cfg())
    resources = list(exp.initial_substrates.keys())
    subs = dict(exp.initial_substrates)
    biomass = float(exp.initial_biomass)

    ts, bs, mus = [], [], []
    S = {r: [] for r in resources}
    U = {r: [] for r in resources}   # CRM uptake predictions u_a
    V = {r: [] for r in resources}   # FBA realized exchange fluxes v_a (negative = uptake)
    # Adaptive CRM: track allocation vector A
    track_A = hasattr(proc.crm, "_A")
    A_hist = {r: [] for r in resources} if track_A else {}

    start = time.time()
    for i in range(exp.steps):
        t = i * exp.dt
        ts.append(t); bs.append(biomass)
        for r in resources:
            S[r].append(subs[r])

        # CRM component: uptake capacities the CRM prescribes this step
        u = proc.crm.compute_uptakes(subs, biomass)
        for r in resources:
            U[r].append(u.get(r, 0.0))

        if track_A:
            for idx, r in enumerate(resources):
                A_hist[r].append(float(proc.crm._A[idx]))

        out = proc.update({"substrates": subs, "biomass": biomass}, exp.dt)

        # Back out realized FBA exchange fluxes (mmol/gDW/hr) from the deltas:
        #   delta_sub = v_a * biomass * dt     =>     v_a = delta_sub / (biomass * dt)
        # Sign convention: v < 0 is uptake, v > 0 is secretion (COBRA convention).
        denom = max(biomass * exp.dt, 1e-12)
        for r in resources:
            V[r].append(out["substrates"].get(r, 0.0) / denom)
        mus.append(out["biomass"] / denom)

        biomass += out["biomass"]
        for r, d in out["substrates"].items():
            subs[r] = max(0.0, subs[r] + d)
    wall = time.time() - start

    return RunResult(
        experiment=exp,
        t=np.array(ts),
        biomass=np.array(bs),
        substrates={r: np.array(v) for r, v in S.items()},
        wall_clock_s=wall,
        crm_uptakes={r: np.array(v) for r, v in U.items()},
        fba_fluxes={r: np.array(v) for r, v in V.items()},
        mu=np.array(mus),
        crm_internal={r: np.array(v) for r, v in A_hist.items()} if track_A else {},
    )


def _simulate_community(exp: Experiment) -> RunResult:
    """
    Multi-consumer community on a shared resource pool.

    exp.extra_meta['species'] is a list of species specs:
        {"name": str, "build_cfg": callable, "initial_biomass": float}

    At each step every species computes its own CRM uptakes from the current
    shared substrate concentrations, runs FBA, and returns its own
    delta-substrates. The shared pool is updated with the *sum* of the
    species' deltas each step. Per-species biomass updates individually.
    """
    species_specs = exp.extra_meta["species"]
    procs = []
    biomass = {}
    for sp in species_specs:
        procs.append((sp["name"], _build_process(sp["build_cfg"]())))
        biomass[sp["name"]] = float(sp["initial_biomass"])

    resources = list(exp.initial_substrates.keys())
    subs = dict(exp.initial_substrates)

    ts = []
    S = {r: [] for r in resources}
    # per-species trajectories
    per_sp: Dict[str, Dict[str, Any]] = {}
    for name, proc in procs:
        per_sp[name] = {
            "biomass": [],
            "mu": [],
            "u": {r: [] for r in resources},
            "v": {r: [] for r in resources},
        }

    start = time.time()
    for i in range(exp.steps):
        t = i * exp.dt
        ts.append(t)
        for r in resources:
            S[r].append(subs[r])

        # Phase 1: per-species CRM uptakes + FBA from shared pool
        pooled_delta_sub = {r: 0.0 for r in resources}
        for name, proc in procs:
            bmass = biomass[name]
            per_sp[name]["biomass"].append(bmass)
            u = proc.crm.compute_uptakes(subs, bmass)
            for r in resources:
                per_sp[name]["u"][r].append(u.get(r, 0.0))

            out = proc.update({"substrates": subs, "biomass": bmass}, exp.dt)
            denom = max(bmass * exp.dt, 1e-12)
            for r in resources:
                delta = out["substrates"].get(r, 0.0)
                per_sp[name]["v"][r].append(delta / denom)
                pooled_delta_sub[r] += delta
            per_sp[name]["mu"].append(out["biomass"] / denom)
            biomass[name] += out["biomass"]

        # Phase 2: shared pool integrates the sum
        for r in resources:
            subs[r] = max(0.0, subs[r] + pooled_delta_sub[r])
    wall = time.time() - start

    # Package: total biomass goes in the standard fields; per-species in `species`
    total_biomass = np.sum(
        [np.array(per_sp[name]["biomass"]) for name in biomass], axis=0
    )
    species_out = {}
    for name in biomass:
        sp = per_sp[name]
        species_out[name] = {
            "biomass": np.array(sp["biomass"]),
            "mu": np.array(sp["mu"]),
            "u": {r: np.array(sp["u"][r]) for r in resources},
            "v": {r: np.array(sp["v"][r]) for r in resources},
        }

    return RunResult(
        experiment=exp,
        t=np.array(ts),
        biomass=total_biomass,
        substrates={r: np.array(v) for r, v in S.items()},
        wall_clock_s=wall,
        species=species_out,
    )


def _simulate_sweep(exp: Experiment) -> RunResult:
    """Custom runner used for the nutrient-sweep experiment."""
    sweep = exp.series_override(exp)
    return RunResult(
        experiment=exp,
        t=sweep["x"],
        biomass=sweep["biomass"],
        substrates={},
        wall_clock_s=sweep["wall_clock_s"],
        extra=sweep.get("extra", {}),
    )


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
def _cfg_from(model_key: str, crm: Dict[str, Any],
              bounds: Optional[Dict] = None,
              substrate_update_reactions: Optional[Dict[str, str]] = None,
              ) -> Dict[str, Any]:
    """Build a CRMDynamicFBA config from a registered GSM plus a CRM spec."""
    spec = get_model_spec(model_key)
    merged_bounds = dict(spec["default_bounds"])
    if bounds:
        merged_bounds.update(bounds)
    return {
        "model_file": spec["model_file"],
        "substrate_update_reactions": substrate_update_reactions or dict(spec["substrate_update_reactions"]),
        "bounds": merged_bounds,
        "biomass_reaction": spec["biomass_reaction"],
        "crm": crm,
        "_model_key": model_key,      # metadata only; passed through for the report
        "_organism": spec["organism"],
    }


def _diauxie_cfg():
    return _cfg_from("ecoli_core", {
        "type": "monod",
        "params": {"kinetic_params": {"glucose": (0.5, 10.0), "acetate": (0.5, 3.0)}},
    })


def _overflow_cfg():
    # Oxygen-limited overflow metabolism (Crabtree-like). Tight O2 cap pushes
    # FBA to secrete acetate rather than fully oxidize glucose.
    return _cfg_from("ecoli_core",
        crm={"type": "macarthur",
             "params": {"c": {"glucose": 1.2, "acetate": 0.4},
                        "resource_mode": "external"}},
        bounds={"EX_o2_e": {"lower": -5.0, "upper": 1000.0}},
    )


def _adaptive_cfg():
    # Tight O2 cap so FBA overflows acetate while glucose is abundant (creates
    # a growing acetate pool to switch to), faster adaptation (lam=5), and a
    # tiny initial acetate seed so r_acetate > 0 at t=0 and the adaptation
    # gradient on A_acetate is non-zero from the start.
    return _cfg_from("ecoli_core",
        crm={
            "type": "adaptive",
            "params": {
                "v": {"glucose": 10.0, "acetate": 5.0},
                "K": {"glucose": 0.5, "acetate": 0.5},
                "lam": 0.6,
                "E_star": 1.0,
                "A0": {"glucose": 0.9, "acetate": 0.1},
                "n_substeps": 50,
            },
        },
        bounds={"EX_o2_e": {"lower": -6.0, "upper": 1000.0}},
    )


def _tilman_cfg():
    return _cfg_from("ecoli_core", {
        "type": "macarthur",
        "params": {"c": {"glucose": 8.0, "acetate": 3.0},
                   "resource_mode": "tilman"},
    })


def _cross_feed_cfg():
    return _cfg_from("ecoli_core",
        crm={"type": "micrm", "params": {"c": {"glucose": 1.0, "acetate": 0.3}}},
        bounds={"EX_o2_e": {"lower": -6.0, "upper": 1000.0}},
    )


def _sweep_series(exp: Experiment) -> Dict[str, Any]:
    """
    Nutrient-sweep: vary initial glucose; report, per glucose0,
      - final biomass
      - apparent yield Δbiomass / glucose consumed
      - peak CRM uptake u_glucose(t=0) (Monod saturates with glucose0)
      - peak FBA realized uptake |v_glucose|  (hits stoichiometric ceiling)
    This makes the CRM → FBA relationship explicit across the sweep.
    """
    start = time.time()
    glucose0_range = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    finals, yields, u_peak, v_peak = [], [], [], []
    for g0 in glucose0_range:
        proc = _build_process(_diauxie_cfg())
        subs = {"glucose": float(g0), "acetate": 0.0}
        biomass = 0.01
        u_max = 0.0
        v_max = 0.0
        for _ in range(exp.steps):
            u = proc.crm.compute_uptakes(subs, biomass).get("glucose", 0.0)
            u_max = max(u_max, u)
            out = proc.update({"substrates": subs, "biomass": biomass}, exp.dt)
            denom = max(biomass * exp.dt, 1e-12)
            v = -out["substrates"].get("glucose", 0.0) / denom  # positive = uptake
            v_max = max(v_max, v)
            biomass += out["biomass"]
            for r, d in out["substrates"].items():
                subs[r] = max(0.0, subs[r] + d)
        finals.append(biomass)
        consumed = g0 - subs["glucose"]
        yields.append((biomass - 0.01) / consumed if consumed > 1e-9 else 0.0)
        u_peak.append(u_max)
        v_peak.append(v_max)
    return {
        "x": glucose0_range,
        "biomass": np.array(finals),
        "wall_clock_s": time.time() - start,
        "extra": {
            "yields": np.array(yields),
            "u_peak": np.array(u_peak),
            "v_peak": np.array(v_peak),
        },
    }


EXPERIMENT_REGISTRY: List[Experiment] = [
    Experiment(
        name="diauxie",
        title="Diauxic shift (Monod)",
        phenomenon="Sequential substrate utilization",
        description=(
            "Classical Monod CRM coupled to the E. coli core GSM. With high initial "
            "glucose, FBA preferentially oxidizes glucose and secretes some acetate. "
            "Once glucose is depleted, the Monod uptake rate on acetate becomes "
            "non-negligible and FBA switches to acetate consumption — the canonical "
            "diauxic shift."
        ),
        build_cfg=_diauxie_cfg,
        initial_substrates={"glucose": 15.0, "acetate": 0.0},
        steps=360,
        dt=0.05,
    ),
    Experiment(
        name="overflow",
        title="Overflow metabolism under O2 limitation (MacArthur)",
        phenomenon="Crabtree-like acetate secretion",
        description=(
            "MacArthur external-resource CRM drives glucose uptake while the oxygen "
            "exchange is tightly capped (EX_o2_e lower = -5). Under O2 limitation the "
            "LP cannot fully oxidize the influx and secretes acetate as overflow — "
            "a Crabtree-like phenotype — even though no acetate was initially present."
        ),
        build_cfg=_overflow_cfg,
        initial_substrates={"glucose": 12.0, "acetate": 0.0},
        steps=300,
        dt=0.05,
    ),
    Experiment(
        name="adaptive",
        title="Adaptive niche switching (Picciani-Mori)",
        phenomenon="Dynamic trait reallocation",
        description=(
            "Adaptive CRM with an internal allocation vector A_a per resource, "
            "constrained by an energy budget E_star. Initial allocation is skewed to "
            "glucose (A0=[0.95, 0.05]) and the oxygen exchange is capped so FBA "
            "secretes acetate as overflow while glucose is abundant. Once glucose "
            "depletes, r_glucose → 0 so the gradient on A_glucose vanishes while "
            "A_acetate keeps climbing under the Picciani-Mori rule — the allocation "
            "vector reallocates toward acetate and growth continues on the acetate "
            "pool that the cell itself produced."
        ),
        build_cfg=_adaptive_cfg,
        initial_substrates={"glucose": 10.0, "acetate": 0.05},
        steps=480,
        dt=0.05,
    ),
    Experiment(
        name="tilman",
        title="Tilman constant-rate consumption (MacArthur)",
        phenomenon="R-independent uptake",
        description=(
            "MacArthur CRM in 'tilman' mode: uptake rate u_a = c_a is independent of "
            "R_a. FBA drives glucose hard and at a constant rate until the "
            "extracellular concentration is exhausted, giving a sharply-cornered "
            "depletion profile characteristic of Tilman-style resource competition."
        ),
        build_cfg=_tilman_cfg,
        initial_substrates={"glucose": 6.0, "acetate": 0.0},
        steps=300,
        dt=0.05,
    ),
    Experiment(
        name="cross_feed",
        title="Self cross-feeding via overflow acetate (MiCRM)",
        phenomenon="Metabolic leakage and re-uptake",
        description=(
            "MiCRM-style mass-action uptake of glucose and acetate on a single "
            "organism. FBA secretes acetate as a byproduct while glucose is abundant "
            "(positive extracellular acetate trajectory); once glucose drops, the "
            "accumulated acetate pool supports a second growth phase — a single-species "
            "analogue of community cross-feeding."
        ),
        build_cfg=_cross_feed_cfg,
        initial_substrates={"glucose": 8.0, "acetate": 0.0},
        steps=360,
        dt=0.05,
    ),
    Experiment(
        name="nutrient_sweep",
        title="Nutrient-limitation sweep",
        phenomenon="Saturating yield curve",
        description=(
            "Sweep over initial glucose (1-80 mmol/L) with the Monod diauxie setup. "
            "Plot final biomass and apparent biomass yield per mmol glucose consumed. "
            "Biomass saturates with resource availability; apparent yield drops at "
            "high glucose as more carbon exits as acetate/CO2 overflow."
        ),
        build_cfg=_diauxie_cfg,
        initial_substrates={"glucose": 0.0, "acetate": 0.0},
        steps=240,
        dt=0.05,
        series_override=_sweep_series,
    ),
    Experiment(
        name="community",
        title="Two-species community on a shared pool",
        phenomenon="Niche differentiation and cross-feeding",
        description=(
            "Two E. coli core GSMs share a single glucose+acetate pool with no "
            "spatial separation. Species 'glucose_specialist' has a strong Monod "
            "preference for glucose (high Vmax_glc, low Vmax_ac) and a tight O2 cap "
            "that forces acetate overflow. Species 'acetate_specialist' has low "
            "glucose affinity and high acetate Vmax. Early on the glucose "
            "specialist dominates, secreting acetate; the acetate specialist then "
            "grows on the acetate byproduct — niche partitioning by cross-feeding, "
            "visible as two offset growth phases on the same GSM."
        ),
        build_cfg=_diauxie_cfg,  # unused; runner_override drives this
        initial_substrates={"glucose": 15.0, "acetate": 0.0},
        steps=360,
        dt=0.05,
        runner_override=_simulate_community,
        extra_meta={
            "species": [
                {
                    "name": "glucose_specialist",
                    "initial_biomass": 0.008,
                    "build_cfg": lambda: _cfg_from(
                        "ecoli_core",
                        crm={"type": "monod",
                             "params": {"kinetic_params": {
                                 "glucose": (0.3, 12.0), "acetate": (2.0, 1.0)}}},
                        bounds={"EX_o2_e": {"lower": -6.0, "upper": 1000.0}},
                    ),
                },
                {
                    "name": "acetate_specialist",
                    "initial_biomass": 0.002,
                    "build_cfg": lambda: _cfg_from(
                        "ecoli_core",
                        crm={"type": "monod",
                             "params": {"kinetic_params": {
                                 "glucose": (5.0, 1.5), "acetate": (0.2, 6.0)}}},
                    ),
                },
            ]
        },
    ),
]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_RES_COLORS = {"glucose": "#2c7fb8", "acetate": "#31a354"}


def _color(resource: str) -> str:
    return _RES_COLORS.get(resource, "#888888")


def _plot_timeseries(result: RunResult, out_path: Path) -> None:
    """
    Four-panel figure per experiment that makes the CRM/FBA coupling explicit:

      (A) Extracellular concentrations + biomass — the system-level trajectory.
      (B) CRM component: u_a(t), the per-resource uptake rate the CRM prescribes.
          This is the quantity the process uses to set FBA exchange lower bounds.
      (C) FBA result: realized exchange flux v_a(t) (negative = uptake,
          positive = secretion) AND the CRM-prescribed uptake envelope |u_a|
          drawn as a dashed line. Gap between v and u shows when FBA hits
          the CRM-imposed cap vs. when it is bounded internally.
      (D) FBA result: realized growth rate μ(t). For CRMs with internal state
          (e.g. Adaptive), that state is overlaid on a twin axis.
    """
    exp = result.experiment
    t = result.t

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    (axA, axB), (axC, axD) = axes

    # ----- (A) biomass + substrates -----
    ax1 = axA
    ax1.plot(t, result.biomass, color="#c0392b", lw=2.2, label="biomass")
    ax1.set_ylabel("biomass (gDW/L)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    for r, vals in result.substrates.items():
        ax2.plot(t, vals, lw=2.0, color=_color(r), label=r)
    ax2.set_ylabel("substrate (mmol/L)")
    ax1.set_title("(A) System trajectory")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9)

    # ----- (B) CRM-prescribed uptake rates u_a(t) -----
    for r, u in result.crm_uptakes.items():
        axB.plot(t, u, lw=2.0, color=_color(r), label=f"u_{r}")
    axB.set_ylabel("CRM uptake u_a\n(mmol/gDW/hr)")
    axB.set_title("(B) CRM component: uptake rates")
    axB.axhline(0, color="#ccc", lw=0.8)
    axB.legend(loc="upper right", frameon=False, fontsize=9)

    # ----- (C) FBA realized fluxes v_a(t) vs CRM envelope -----
    for r, v in result.fba_fluxes.items():
        col = _color(r)
        axC.plot(t, v, lw=2.0, color=col, label=f"v_{r} (FBA)")
        if r in result.crm_uptakes:
            # CRM imposes EX lower bound = -u_a; show ±u_a envelope
            axC.plot(t, -result.crm_uptakes[r], lw=1.2, ls="--", color=col, alpha=0.7,
                     label=f"-u_{r} (CRM bound)")
    axC.axhline(0, color="#888", lw=0.8)
    axC.set_xlabel("time (hr)")
    axC.set_ylabel("exchange flux\n(mmol/gDW/hr)")
    axC.set_title("(C) FBA result: realized fluxes vs CRM bound")
    axC.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)

    # ----- (D) growth rate + optional CRM internal state -----
    axD.plot(t, result.mu if result.mu is not None else np.zeros_like(t),
             color="#c0392b", lw=2.2, label="μ (FBA)")
    axD.set_ylabel("growth rate μ (1/hr)", color="#c0392b")
    axD.tick_params(axis="y", labelcolor="#c0392b")
    axD.set_xlabel("time (hr)")
    axD.set_title("(D) FBA growth rate" +
                  (" + CRM allocation" if result.crm_internal else ""))
    if result.crm_internal:
        axD2 = axD.twinx()
        for r, A in result.crm_internal.items():
            axD2.plot(t, A, lw=1.8, ls=":", color=_color(r), label=f"A_{r}")
        axD2.set_ylabel("CRM allocation A_a")
        h1, l1 = axD.get_legend_handles_labels()
        h2, l2 = axD2.get_legend_handles_labels()
        axD.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=9)
    else:
        axD.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(exp.title, fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_sweep(result: RunResult, out_path: Path) -> None:
    """
    Three-panel sweep figure:
      (A) Final biomass and apparent yield vs initial glucose.
      (B) CRM component: peak u_glucose saturates (Monod Vmax).
      (C) FBA response: peak realized uptake |v_glucose| hits a ceiling set
          by stoichiometry/O2, even as the CRM keeps demanding more.
          The gap between (B) and (C) is the CRM→FBA coupling becoming slack.
    """
    yields = result.extra.get("yields")
    u_peak = result.extra.get("u_peak")
    v_peak = result.extra.get("v_peak")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axA.plot(result.t, result.biomass, marker="o", color="#c0392b", lw=2.0,
             label="final biomass")
    axA.set_xlabel("initial glucose (mmol/L)")
    axA.set_ylabel("final biomass (gDW/L)", color="#c0392b")
    axA.tick_params(axis="y", labelcolor="#c0392b")
    axA.set_xscale("log")
    axA.set_title("(A) System trajectory")
    if yields is not None:
        axA2 = axA.twinx()
        axA2.plot(result.t, yields, marker="s", color="#2c7fb8", lw=2.0,
                  label="apparent yield")
        axA2.set_ylabel("Δbiomass / glucose consumed", color="#2c7fb8")
        axA2.tick_params(axis="y", labelcolor="#2c7fb8")

    if u_peak is not None:
        axB.plot(result.t, u_peak, marker="o", color=_color("glucose"), lw=2.0)
        axB.set_xscale("log")
        axB.set_xlabel("initial glucose (mmol/L)")
        axB.set_ylabel("peak u_glucose (mmol/gDW/hr)")
        axB.set_title("(B) CRM component")

    if v_peak is not None:
        axC.plot(result.t, v_peak, marker="s", color=_color("glucose"), lw=2.0,
                 label="|v_glucose| (FBA)")
        if u_peak is not None:
            axC.plot(result.t, u_peak, ls="--", lw=1.3, color=_color("glucose"),
                     alpha=0.7, label="u_glucose (CRM)")
        axC.set_xscale("log")
        axC.set_xlabel("initial glucose (mmol/L)")
        axC.set_ylabel("peak uptake (mmol/gDW/hr)")
        axC.set_title("(C) FBA result vs CRM bound")
        axC.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle(result.experiment.title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


_SPECIES_COLORS = ["#8856a7", "#e6550d", "#1b7837", "#0868ac"]


def _plot_community(result: RunResult, out_path: Path) -> None:
    """
    Four panels for multi-species community runs:
      (A) per-species biomass + shared substrate pool
      (B) CRM component: u_a(t) per species per resource — shows that each
          species' CRM prescribes different uptake rates from the *same* pool
      (C) FBA result: v_a(t) per species per resource; positive = secretion
          (producer), negative = uptake (consumer). Cross-feeding is visible
          when one species' v_a is positive while another's is negative on
          the same resource.
      (D) per-species growth rate μ_i(t) — offset growth peaks reveal niche
          partitioning.
    """
    exp = result.experiment
    t = result.t
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    (axA, axB), (axC, axD) = axes

    # ----- (A) per-species biomass + shared pool -----
    for i, (name, data) in enumerate(result.species.items()):
        axA.plot(t, data["biomass"], lw=2.2,
                 color=_SPECIES_COLORS[i % len(_SPECIES_COLORS)],
                 label=f"{name} biomass")
    axA.set_ylabel("biomass (gDW/L)")
    ax2 = axA.twinx()
    for r, vals in result.substrates.items():
        ax2.plot(t, vals, lw=1.8, ls="--", color=_color(r), label=f"{r} (shared)")
    ax2.set_ylabel("substrate (mmol/L)")
    axA.set_title("(A) Per-species biomass + shared resource pool")
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8)

    # ----- (B) per-species CRM uptake rates -----
    for i, (name, data) in enumerate(result.species.items()):
        sp_col = _SPECIES_COLORS[i % len(_SPECIES_COLORS)]
        for r, arr in data["u"].items():
            axB.plot(t, arr, lw=1.8, color=sp_col,
                     ls="-" if r == "glucose" else ":",
                     label=f"{name} · u_{r}")
    axB.set_ylabel("CRM uptake u_a (mmol/gDW/hr)")
    axB.set_title("(B) CRM component: per-species uptake rates")
    axB.legend(loc="upper right", frameon=False, fontsize=7)

    # ----- (C) per-species FBA realized fluxes -----
    for i, (name, data) in enumerate(result.species.items()):
        sp_col = _SPECIES_COLORS[i % len(_SPECIES_COLORS)]
        for r, arr in data["v"].items():
            axC.plot(t, arr, lw=1.8, color=sp_col,
                     ls="-" if r == "glucose" else ":",
                     label=f"{name} · v_{r}")
    axC.axhline(0, color="#888", lw=0.8)
    axC.set_ylabel("exchange flux v_a (mmol/gDW/hr)")
    axC.set_xlabel("time (hr)")
    axC.set_title("(C) FBA result: realized fluxes (>0 secrete, <0 uptake)")
    axC.legend(loc="upper right", frameon=False, fontsize=7)

    # ----- (D) per-species growth rate -----
    for i, (name, data) in enumerate(result.species.items()):
        axD.plot(t, data["mu"], lw=2.0,
                 color=_SPECIES_COLORS[i % len(_SPECIES_COLORS)],
                 label=f"μ({name})")
    axD.set_ylabel("growth rate μ_i (1/hr)")
    axD.set_xlabel("time (hr)")
    axD.set_title("(D) FBA growth rate per species")
    axD.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(exp.title, fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_for(result: RunResult, out_path: Path) -> None:
    if result.species is not None:
        _plot_community(result, out_path)
    elif result.experiment.series_override is not None:
        _plot_sweep(result, out_path)
    else:
        _plot_timeseries(result, out_path)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
_STYLE = """
  body { font-family: system-ui, sans-serif; max-width: 1020px; margin: 2rem auto; background: #fafafa; color: #222; padding: 0 1rem; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .3rem; }
  h3 { margin-top: 1.2rem; margin-bottom: 0.4rem; }
  section { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }
  h2 { margin-top: 0; }
  table { border-collapse: collapse; margin: .8rem 0; width: 100%; }
  td { padding: .3rem .8rem; border: 1px solid #eee; vertical-align: top; }
  td:first-child { font-weight: 600; width: 200px; }
  img { max-width: 100%; margin-top: .5rem; border: 1px solid #ddd; border-radius: 4px; }
  .pill { display: inline-block; padding: 2px 10px; border: 1px solid #b8c7dc; border-radius: 999px; background: #eef3fa; color: #0366d6; font-size: 12px; margin-right: 6px; }
  .meta { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 1rem 0; font-size: 13px; color: #555; }
  .meta div { display: inline-block; margin-right: 1.5rem; }
  .meta code { background: #f0f0f0; padding: 1px 6px; border-radius: 4px; }
  .meta a { color: #0366d6; text-decoration: none; }
  .experiment-nav { position: sticky; top: 0; z-index: 10; background: #fff; border: 1px solid #ddd; border-radius: 8px;
                    padding: 0.6rem 1rem; margin: 1rem 0; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
  .experiment-nav .nav-title { font-weight: 600; margin-right: 0.4rem; color: #555; }
  .experiment-nav a { padding: 3px 10px; border: 1px solid #ddd; border-radius: 999px; background: #fafafa;
                      color: #0366d6; text-decoration: none; font-size: 13px; }
  .experiment-nav a:hover { background: #eef3fa; border-color: #b8c7dc; }
  section { scroll-margin-top: 4rem; }
  pre.config { background: #f7f7f7; border: 1px solid #eee; border-radius: 6px; padding: 0.6rem 0.8rem;
               font-size: 12px; overflow-x: auto; }
"""


def _short_cfg(cfg: Dict[str, Any]) -> str:
    crm = cfg.get("crm", {})
    params = crm.get("params", {})
    return (
        f"model_file        = {cfg.get('model_file')!r}   # {cfg.get('_organism', '')}\n"
        f"biomass_reaction  = {cfg.get('biomass_reaction')!r}\n"
        f"crm.type          = {crm.get('type')}\n"
        f"crm.params        = {params}\n"
        f"substrate_update_reactions = {cfg.get('substrate_update_reactions')}\n"
        f"bounds            = {cfg.get('bounds')}"
    )


def _gather_meta() -> Dict[str, str]:
    def _git(cmd):
        try:
            return subprocess.check_output(["git"] + cmd, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    commit = _git(["rev-parse", "HEAD"])
    remote = _git(["config", "--get", "remote.origin.url"])
    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote[len("git@github.com:"):]
    return {
        "generated": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "host": socket.gethostname(),
        "commit": commit,
        "remote": remote,
    }


def _section_html(r: RunResult, doc_dir: Path) -> str:
    exp = r.experiment
    img_rel = f"{exp.name}.png"

    # Collect GSM info: may be one (single) or many (community) configs
    if exp.runner_override is _simulate_community:
        species_cfgs = [(sp["name"], sp["build_cfg"]()) for sp in exp.extra_meta["species"]]
        cfg = species_cfgs[0][1]
    else:
        species_cfgs = [("", exp.build_cfg())]
        cfg = species_cfgs[0][1]

    gsm_rows = []
    for sp_name, sp_cfg in species_cfgs:
        label = f" ({sp_name})" if sp_name else ""
        gsm_rows.append(
            f"<tr><td>GSM{label}</td>"
            f"<td><code>{sp_cfg.get('model_file')}</code> — "
            f"{_html.escape(str(sp_cfg.get('_organism', '')))}<br>"
            f"<span style='font-size:12px;color:#666'>biomass: "
            f"<code>{sp_cfg.get('biomass_reaction')}</code></span></td></tr>"
        )
    crm_type = cfg["crm"]["type"]

    final_rows = [
        f"<tr><td>Phenomenon</td><td>{_html.escape(exp.phenomenon)}</td></tr>",
        *gsm_rows,
        f"<tr><td>CRM type</td><td><code>{crm_type}</code></td></tr>",
        f"<tr><td>Steps × dt</td><td>{exp.steps} × {exp.dt} hr</td></tr>",
        f"<tr><td>Wall-clock time</td><td>{r.wall_clock_s:.2f} s</td></tr>",
    ]
    if r.species is not None:
        for sp_name, data in r.species.items():
            final_rows.append(
                f"<tr><td>Final biomass ({sp_name})</td>"
                f"<td>{data['biomass'][-1]:.3f} gDW/L &nbsp; "
                f"peak μ = {float(np.max(data['mu'])):.3f} 1/hr</td></tr>"
            )
        for res, vals in r.substrates.items():
            final_rows.append(
                f"<tr><td>Final {res} (shared pool)</td>"
                f"<td>{vals[-1]:.3f} mmol/L</td></tr>"
            )
    elif r.substrates:
        final_rows.append(
            f"<tr><td>Final biomass</td><td>{r.biomass[-1]:.3f} gDW/L</td></tr>"
        )
        for res, vals in r.substrates.items():
            final_rows.append(
                f"<tr><td>Final {res}</td><td>{vals[-1]:.3f} mmol/L</td></tr>"
            )
        if r.mu is not None and r.mu.size:
            final_rows.append(
                f"<tr><td>Peak μ (FBA)</td><td>{np.max(r.mu):.3f} 1/hr</td></tr>"
            )
        for res, u in r.crm_uptakes.items():
            peak_u = float(np.max(u)) if u.size else 0.0
            peak_v = float(np.max(-r.fba_fluxes[res])) if r.fba_fluxes.get(res) is not None and r.fba_fluxes[res].size else 0.0
            final_rows.append(
                f"<tr><td>Peak {res} CRM→FBA</td>"
                f"<td>u={peak_u:.2f} mmol/gDW/hr &nbsp; |v|={peak_v:.2f} mmol/gDW/hr</td></tr>"
            )
    else:
        final_rows.append(
            f"<tr><td>Range</td><td>glucose0 ∈ [{r.t.min():.1f}, {r.t.max():.1f}] mmol/L</td></tr>"
        )

    cfg_blocks = "\n\n".join(
        (f"# {sp_name}\n{_short_cfg(sp_cfg)}" if sp_name else _short_cfg(sp_cfg))
        for sp_name, sp_cfg in species_cfgs
    )

    return f"""
  <section id="{exp.name}">
    <h2>{_html.escape(exp.title)}</h2>
    <span class="pill">{_html.escape(exp.phenomenon)}</span>
    <span class="pill">CRM: {crm_type}</span>
    <p>{_html.escape(exp.description)}</p>
    <h3>Result</h3>
    <img src="{img_rel}" alt="{exp.name}" />
    <h3>Summary</h3>
    <table>{''.join(final_rows)}</table>
    <h3>Configuration</h3>
    <pre class="config">{_html.escape(cfg_blocks)}</pre>
  </section>
"""


def generate_html_report(results: List[RunResult], doc_dir: Path) -> Path:
    doc_dir.mkdir(parents=True, exist_ok=True)
    meta = _gather_meta()
    commit_html = ""
    if meta["commit"] and meta["remote"]:
        short = meta["commit"][:8]
        commit_html = (
            f'<div><strong>Commit:</strong> '
            f'<a href="{meta["remote"]}/commit/{meta["commit"]}">'
            f'<code>{short}</code></a></div>'
        )
    nav_links = "".join(
        f'<a href="#{r.experiment.name}">{_html.escape(r.experiment.title)}</a>'
        for r in results
    )
    sections = "".join(_section_html(r, doc_dir) for r in results)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CRM-FBA experiments</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>CRM-FBA experiments</h1>
<p>A suite of CRM-coupled dynamic FBA simulations on the <em>E. coli</em>
core GSM, illustrating ecological and microbial-metabolic phenomena through
different Consumer Resource Model / FBA couplings.</p>
<p>Each experiment plot makes the coupling explicit with four panels:
<strong>(A)</strong> the system trajectory (biomass + extracellular resources);
<strong>(B)</strong> the <em>CRM component</em>, i.e. the per-resource uptake
rate <code>u_a(t)</code> that the CRM prescribes and which the process writes
into <code>EX_rxn.lower_bound = −u_a</code>;
<strong>(C)</strong> the <em>FBA result</em>, the realized exchange flux
<code>v_a(t)</code> (negative = uptake, positive = secretion) plotted against
the CRM-imposed envelope <code>−u_a</code> — where the two curves touch, FBA
is CRM-limited; where they separate, FBA is stoichiometry- or O₂-limited;
<strong>(D)</strong> the FBA-realized growth rate <code>μ(t)</code> (plus any
CRM-internal state such as Adaptive's allocation vector <code>A_a</code>).</p>
<div class="meta">
  <div><strong>Generated:</strong> {meta['generated']}</div>
  <div><strong>On:</strong> {_html.escape(meta['host'])}</div>
  {commit_html}
</div>
<nav class="experiment-nav">
  <span class="nav-title">Experiments:</span>
  {nav_links}
</nav>
{sections}
</body>
</html>
"""
    out = doc_dir / "index.html"
    out.write_text(html)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_all(only: Optional[List[str]] = None) -> List[RunResult]:
    selected = [e for e in EXPERIMENT_REGISTRY if not only or e.name in only]
    results = []
    for exp in selected:
        print(f"[run] {exp.name}: {exp.title}")
        if exp.runner_override is not None:
            result = exp.runner_override(exp)
        elif exp.series_override is not None:
            result = _simulate_sweep(exp)
        else:
            result = _simulate(exp)
        img_path = DOC_DIR / f"{exp.name}.png"
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        _plot_for(result, img_path)
        print(f"       wall={result.wall_clock_s:.2f}s  plot={img_path.relative_to(REPO_ROOT)}")
        results.append(result)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run CRM-FBA experiment suite.")
    parser.add_argument("--no-open", action="store_true", help="don't open the report in a browser")
    parser.add_argument("--only", nargs="*", help="run only the listed experiment names")
    args = parser.parse_args(argv)

    results = run_all(args.only)
    report = generate_html_report(results, DOC_DIR)
    print(f"\nReport written to {report}")

    if not args.no_open:
        webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
