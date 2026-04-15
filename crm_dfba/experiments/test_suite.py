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


@dataclass
class RunResult:
    experiment: Experiment
    t: np.ndarray
    biomass: np.ndarray
    substrates: Dict[str, np.ndarray]
    wall_clock_s: float
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------
def _build_process(cfg):
    return CRMDynamicFBA(config=cfg, core=_core)


def _simulate(exp: Experiment) -> RunResult:
    proc = _build_process(exp.build_cfg())
    subs = dict(exp.initial_substrates)
    biomass = float(exp.initial_biomass)

    ts, bs = [], []
    S = {r: [] for r in subs}

    start = time.time()
    for i in range(exp.steps):
        t = i * exp.dt
        ts.append(t); bs.append(biomass)
        for r in subs:
            S[r].append(subs[r])
        out = proc.update({"substrates": subs, "biomass": biomass}, exp.dt)
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
ECOLI_CORE_SUBSTRATE_RXNS = {"glucose": "EX_glc__D_e", "acetate": "EX_ac_e"}
ECOLI_CORE_BIOMASS = "Biomass_Ecoli_core"


def _ecoli_core_cfg(crm, bounds=None):
    return {
        "model_file": "textbook",
        "substrate_update_reactions": ECOLI_CORE_SUBSTRATE_RXNS,
        "bounds": bounds or {"EX_o2_e": {"lower": -20.0, "upper": 1000.0},
                             "ATPM": {"lower": 1.0, "upper": 1.0}},
        "biomass_reaction": ECOLI_CORE_BIOMASS,
        "crm": crm,
    }


def _diauxie_cfg():
    return _ecoli_core_cfg({
        "type": "monod",
        "params": {"kinetic_params": {"glucose": (0.5, 10.0), "acetate": (0.5, 3.0)}},
    })


def _overflow_cfg():
    # Oxygen-limited overflow metabolism (Crabtree-like). Tight O2 cap pushes
    # FBA to secrete acetate rather than fully oxidize glucose.
    return _ecoli_core_cfg(
        {"type": "macarthur",
         "params": {"c": {"glucose": 1.2, "acetate": 0.4},
                    "resource_mode": "external"}},
        bounds={"EX_o2_e": {"lower": -5.0, "upper": 1000.0},
                "ATPM": {"lower": 1.0, "upper": 1.0}},
    )


def _adaptive_cfg():
    return _ecoli_core_cfg({
        "type": "adaptive",
        "params": {
            "v": {"glucose": 12.0, "acetate": 6.0},
            "K": {"glucose": 0.5, "acetate": 0.5},
            "lam": 1.2,
            "E_star": 1.0,
            "A0": {"glucose": 0.9, "acetate": 0.1},
        },
    })


def _tilman_cfg():
    return _ecoli_core_cfg({
        "type": "macarthur",
        "params": {"c": {"glucose": 8.0, "acetate": 3.0},
                   "resource_mode": "tilman"},
    })


def _cross_feed_cfg():
    return _ecoli_core_cfg({
        "type": "micrm",
        "params": {"c": {"glucose": 1.0, "acetate": 0.3}},
    })


def _sweep_series(exp: Experiment) -> Dict[str, Any]:
    """Nutrient-sweep: vary initial glucose, report final biomass + yield."""
    start = time.time()
    glucose0_range = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    finals = []
    yields = []
    for g0 in glucose0_range:
        proc = _build_process(_diauxie_cfg())
        subs = {"glucose": float(g0), "acetate": 0.0}
        biomass = 0.01
        for _ in range(exp.steps):
            out = proc.update({"substrates": subs, "biomass": biomass}, exp.dt)
            biomass += out["biomass"]
            for r, d in out["substrates"].items():
                subs[r] = max(0.0, subs[r] + d)
        finals.append(biomass)
        # apparent yield = Δbiomass / glucose consumed
        consumed = g0 - subs["glucose"]
        yields.append((biomass - 0.01) / consumed if consumed > 1e-9 else 0.0)
    return {
        "x": glucose0_range,
        "biomass": np.array(finals),
        "wall_clock_s": time.time() - start,
        "extra": {"yields": np.array(yields)},
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
            "glucose (A0=[0.9, 0.1]). As glucose depletes and acetate accumulates from "
            "FBA overflow, A reallocates toward acetate under the Picciani-Mori "
            "adaptation dynamics."
        ),
        build_cfg=_adaptive_cfg,
        initial_substrates={"glucose": 10.0, "acetate": 0.0},
        steps=360,
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
]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_timeseries(result: RunResult, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(result.t, result.biomass, color="#c0392b", lw=2.2, label="biomass")
    ax1.set_xlabel("time (hr)")
    ax1.set_ylabel("biomass (gDW/L)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    ax2 = ax1.twinx()
    colors = {"glucose": "#2c7fb8", "acetate": "#31a354"}
    for r, vals in result.substrates.items():
        ax2.plot(result.t, vals, lw=2.0, color=colors.get(r, "#888"), label=r)
    ax2.set_ylabel("substrate (mmol/L)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)

    ax1.set_title(result.experiment.title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_sweep(result: RunResult, out_path: Path) -> None:
    yields = result.extra.get("yields")
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(result.t, result.biomass, marker="o", color="#c0392b", lw=2.0, label="final biomass")
    ax1.set_xlabel("initial glucose (mmol/L)")
    ax1.set_ylabel("final biomass (gDW/L)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_xscale("log")

    if yields is not None:
        ax2 = ax1.twinx()
        ax2.plot(result.t, yields, marker="s", color="#2c7fb8", lw=2.0, label="apparent yield")
        ax2.set_ylabel("yield Δbiomass / glucose consumed", color="#2c7fb8")
        ax2.tick_params(axis="y", labelcolor="#2c7fb8")

    ax1.set_title(result.experiment.title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_for(result: RunResult, out_path: Path) -> None:
    if result.experiment.series_override is not None:
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
        f"crm.type  = {crm.get('type')}\n"
        f"crm.params = {params}\n"
        f"substrate_update_reactions = {cfg.get('substrate_update_reactions')}\n"
        f"bounds = {cfg.get('bounds')}"
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
    cfg = exp.build_cfg()
    crm_type = cfg["crm"]["type"]

    final_rows = [
        f"<tr><td>Phenomenon</td><td>{_html.escape(exp.phenomenon)}</td></tr>",
        f"<tr><td>CRM type</td><td><code>{crm_type}</code></td></tr>",
        f"<tr><td>Steps × dt</td><td>{exp.steps} × {exp.dt} hr</td></tr>",
        f"<tr><td>Wall-clock time</td><td>{r.wall_clock_s:.2f} s</td></tr>",
    ]
    if r.substrates:
        final_rows.append(
            f"<tr><td>Final biomass</td><td>{r.biomass[-1]:.3f} gDW/L</td></tr>"
        )
        for res, vals in r.substrates.items():
            final_rows.append(
                f"<tr><td>Final {res}</td><td>{vals[-1]:.3f} mmol/L</td></tr>"
            )
    else:
        final_rows.append(
            f"<tr><td>Range</td><td>glucose0 ∈ [{r.t.min():.1f}, {r.t.max():.1f}] mmol/L</td></tr>"
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
    <pre class="config">{_html.escape(_short_cfg(cfg))}</pre>
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
different Consumer Resource Model / FBA couplings. The CRM computes
per-resource uptake rates; these become the FBA exchange lower bounds, and
FBA resolves μ and realized fluxes under stoichiometry.</p>
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
        if exp.series_override is not None:
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
