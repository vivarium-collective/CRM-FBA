"""
Verify the CRM-FBA composite reproduces the monolithic process
trajectory-for-trajectory, across every CRM variant.
"""
from __future__ import annotations

import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from crm_dfba import CRMDynamicFBAMonolithic, crm_dfba_spec
from crm_dfba.demo import CRM_CONFIGS, SUBSTRATE_RXNS, STATIC_BOUNDS


BASE_CONFIG = {
    "model_file": "textbook",
    "substrate_update_reactions": SUBSTRATE_RXNS,
    "bounds": STATIC_BOUNDS,
    "biomass_reaction": "Biomass_Ecoli_core",
}
INITIAL_SUBSTRATES = {"glucose": 11.1, "acetate": 0.0}
INITIAL_BIOMASS = 0.01
DT = 0.05
STEPS = 40
TOL = 1e-9


def run_monolith(crm_cfg):
    core = allocate_core()
    proc = CRMDynamicFBAMonolithic(
        config={**BASE_CONFIG, "crm": crm_cfg}, core=core
    )
    subs = dict(INITIAL_SUBSTRATES)
    X = float(INITIAL_BIOMASS)
    traj = []
    for i in range(STEPS):
        traj.append((i * DT, X, dict(subs)))
        out = proc.update({"substrates": subs, "biomass": X}, DT)
        X += out["biomass"]
        for r, d in out["substrates"].items():
            subs[r] = max(0.0, subs[r] + d)
    return traj


def run_composite(crm_cfg):
    core = allocate_core()
    config = {**BASE_CONFIG, "crm": crm_cfg}
    state = {
        "substrates": dict(INITIAL_SUBSTRATES),
        "biomass": float(INITIAL_BIOMASS),
        "uptakes": {r: 0.0 for r in INITIAL_SUBSTRATES},
        "interval": float(DT),
        "emitter": emitter_from_wires({
            "global_time": ["global_time"],
            "biomass": ["biomass"],
            "substrates": ["substrates"],
        }),
    }
    state.update(crm_dfba_spec(config, dt=DT))
    doc = {
        "schema": {
            "substrates": "map[concentration]",
            "biomass": "mass",
            "uptakes": "overwrite[map[float]]",
            "interval": "overwrite[float]",
        },
        "state": state,
    }
    composite = Composite(doc, core=core)
    composite.run(STEPS * DT)
    history = gather_emitter_results(composite)[("emitter",)]
    return [(h["global_time"], h["biomass"], dict(h["substrates"])) for h in history]


@pytest.mark.parametrize("name", list(CRM_CONFIGS.keys()))
def test_composite_matches_monolith(name):
    mono = run_monolith(CRM_CONFIGS[name])
    comp = run_composite(CRM_CONFIGS[name])
    assert len(mono) == len(comp), f"trajectory lengths differ: {len(mono)} vs {len(comp)}"
    for (tm, xm, sm), (tc, xc, sc) in zip(mono, comp):
        assert abs(tm - tc) < TOL, f"time mismatch: {tm} vs {tc}"
        assert abs(xm - xc) < TOL, f"biomass mismatch at t={tm}: {xm} vs {xc}"
        for r in sm:
            assert abs(sm[r] - sc[r]) < TOL, (
                f"{r} mismatch at t={tm}: {sm[r]} vs {sc[r]}"
            )
