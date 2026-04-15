"""
Minimal demo: run the CRM-FBA composite on ecoli core with each CRM
variant.

Builds a process-bigraph document wiring CRMProcess -> FBAStep via
shared stores for substrates/biomass/uptakes/interval, plus a RAM
emitter for trajectories. The document is handed to Composite.run.
"""
from __future__ import annotations
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from crm_dfba import crm_dfba_spec


RESOURCES = ("glucose", "acetate")
SUBSTRATE_RXNS = {"glucose": "EX_glc__D_e", "acetate": "EX_ac_e"}
STATIC_BOUNDS = {
    "EX_o2_e": {"lower": -20.0, "upper": 1000.0},
    "ATPM": {"lower": 1.0, "upper": 1.0},
}


def build_document(crm_cfg, dt, initial_substrates, initial_biomass):
    config = {
        "model_file": "textbook",
        "substrate_update_reactions": SUBSTRATE_RXNS,
        "bounds": STATIC_BOUNDS,
        "biomass_reaction": "Biomass_Ecoli_core",
        "crm": crm_cfg,
    }
    state = {
        "substrates": dict(initial_substrates),
        "biomass": float(initial_biomass),
        "uptakes": {r: 0.0 for r in initial_substrates},
        "interval": float(dt),
        "emitter": emitter_from_wires({
            "global_time": ["global_time"],
            "biomass": ["biomass"],
            "substrates": ["substrates"],
        }),
    }
    state.update(crm_dfba_spec(config, dt=dt))
    return {
        "schema": {
            "substrates": "map[concentration]",
            "biomass": "mass",
            "uptakes": "overwrite[map[float]]",
            "interval": "overwrite[float]",
        },
        "state": state,
    }


def run_variant(crm_cfg, steps=100, dt=0.05):
    core = allocate_core()
    doc = build_document(
        crm_cfg,
        dt=dt,
        initial_substrates={"glucose": 11.1, "acetate": 0.0},
        initial_biomass=0.01,
    )
    composite = Composite(doc, core=core)
    composite.run(steps * dt)
    return gather_emitter_results(composite)[("emitter",)]


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
        history = run_variant(cfg, steps=100, dt=0.05)
        final = history[-1]
        subs = final.get("substrates", {})
        print(
            f"[{name:>9}] t={final.get('global_time'):.2f}  "
            f"biomass={final.get('biomass'):.4g}  "
            f"glucose={subs.get('glucose', 0.0):.4g}  "
            f"acetate={subs.get('acetate', 0.0):.4g}"
        )


if __name__ == "__main__":
    main()
