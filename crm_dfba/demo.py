"""
Minimal demo: run CRMDynamicFBA on ecoli core with each CRM variant.

Uses a simple manual stepping loop (no Composite) to keep the demo
dependency-light; the process itself is still a process_bigraph Process.
"""
from __future__ import annotations
import numpy as np
from process_bigraph import allocate_core

from crm_dfba import CRMDynamicFBA


_core = allocate_core()


RESOURCES = ("glucose", "acetate")
SUBSTRATE_RXNS = {"glucose": "EX_glc__D_e", "acetate": "EX_ac_e"}
STATIC_BOUNDS = {
    "EX_o2_e": {"lower": -20.0, "upper": 1000.0},
    "ATPM": {"lower": 1.0, "upper": 1.0},
}


def build(crm_cfg):
    return CRMDynamicFBA(
        config={
            "model_file": "textbook",
            "substrate_update_reactions": SUBSTRATE_RXNS,
            "bounds": STATIC_BOUNDS,
            "biomass_reaction": "Biomass_Ecoli_core",
            "crm": crm_cfg,
        },
        core=_core,
    )


def run(proc, steps=200, dt=0.05):
    substrates = {"glucose": 11.1, "acetate": 0.0}
    biomass = 0.01
    traj = {"t": [], "biomass": [], **{r: [] for r in RESOURCES}}
    for i in range(steps):
        t = i * dt
        traj["t"].append(t)
        traj["biomass"].append(biomass)
        for r in RESOURCES:
            traj[r].append(substrates[r])
        out = proc.update({"substrates": substrates, "biomass": biomass}, dt)
        biomass += out["biomass"]
        for r, d in out["substrates"].items():
            substrates[r] = max(0.0, substrates[r] + d)
    return traj


CRM_CONFIGS = {
    "monod": {
        "type": "monod",
        "params": {"kinetic_params": {"glucose": (0.5, 10.0), "acetate": (0.5, 2.0)}},
    },
    "macarthur": {
        "type": "macarthur",
        "params": {
            "c": {"glucose": 0.9, "acetate": 0.2},
            "resource_mode": "external",
        },
    },
    "mcrm": {
        "type": "mcrm",
        "params": {"C": {"glucose": 0.9, "acetate": 0.2}},
    },
    "micrm": {
        "type": "micrm",
        "params": {"c": {"glucose": 0.9, "acetate": 0.2}},
    },
    "adaptive": {
        "type": "adaptive",
        "params": {
            "v": {"glucose": 10.0, "acetate": 5.0},
            "K": {"glucose": 0.5, "acetate": 0.5},
            "lam": 0.5,
            "E_star": 1.0,
        },
    },
}


def main():
    for name, cfg in CRM_CONFIGS.items():
        proc = build(cfg)
        traj = run(proc, steps=100, dt=0.05)
        final = {
            "biomass": traj["biomass"][-1],
            "glucose": traj["glucose"][-1],
            "acetate": traj["acetate"][-1],
        }
        print(f"[{name:>9}] final {final}")


if __name__ == "__main__":
    main()
