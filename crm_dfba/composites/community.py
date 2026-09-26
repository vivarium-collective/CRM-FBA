"""Two-species E. coli community composite.

The richest CRM-FBA bigraph in the workspace: two E. coli core GSMs share a
single glucose + acetate pool with no spatial separation. Each species is its
own ``CRMProcess -> FBAStep`` pair writing the *same* ``substrates`` store, so
one species' overflow becomes the other's feedstock.

- ``glucose_specialist`` — strong Monod preference for glucose (high Vmax_glc,
  low Vmax_ac) with a tight O2 cap (EX_o2_e lower = -6), so it overflows
  acetate while growing fast.
- ``acetate_specialist`` — low glucose affinity, high acetate Vmax, so it grows
  on the acetate the first species secretes.

The result is niche differentiation by cross-feeding: two offset growth phases
on a shared pool. Wired here as a ``@composite_generator`` so the workbench can
browse, run and drill into it.
"""
from __future__ import annotations

from process_bigraph.emitter import emitter_from_wires

from crm_dfba.models import get_model_spec
from crm_dfba.processes.crm_dfba import crm_dfba_spec


_MODEL_KEY = "ecoli_core"
_INITIAL_SUBSTRATES = {"glucose": 15.0, "acetate": 0.0}
_DT = 0.05

# Two Monod specialists on the same GSM. kinetic_params are (Km, Vmax) per
# resource; the O2 cap on the glucose specialist forces acetate overflow.
_SPECIES = [
    {
        "name": "glucose_specialist",
        "initial_biomass": 0.008,
        "crm": {
            "type": "monod",
            "params": {"kinetic_params": {"glucose": (0.3, 12.0), "acetate": (2.0, 1.0)}},
        },
        "bounds": {"EX_o2_e": {"lower": -6.0, "upper": 1000.0}},
    },
    {
        "name": "acetate_specialist",
        "initial_biomass": 0.002,
        "crm": {
            "type": "monod",
            "params": {"kinetic_params": {"glucose": (5.0, 1.5), "acetate": (0.2, 6.0)}},
        },
        "bounds": {},
    },
]


def _species_cfg(sp: dict) -> dict:
    spec = get_model_spec(_MODEL_KEY)
    bounds = dict(spec["default_bounds"])
    bounds.update(sp.get("bounds") or {})
    return {
        "model_file": spec["model_file"],
        "substrate_update_reactions": dict(spec["substrate_update_reactions"]),
        "bounds": bounds,
        "biomass_reaction": spec["biomass_reaction"],
        "crm": sp["crm"],
    }


def community_document(core=None) -> dict:
    """Build the two-species community process-bigraph document."""
    resources = list(get_model_spec(_MODEL_KEY)["substrate_update_reactions"].keys())

    state: dict = {
        "substrates": {r: float(_INITIAL_SUBSTRATES.get(r, 0.0)) for r in resources},
    }
    schema: dict = {"substrates": "map[concentration]"}
    emit_wires: dict = {"global_time": ["global_time"], "substrates": ["substrates"]}

    for sp in _SPECIES:
        name = sp["name"]
        bio, up, iv = f"biomass_{name}", f"uptakes_{name}", f"interval_{name}"
        state[bio] = float(sp["initial_biomass"])
        state[up] = {r: 0.0 for r in resources}
        state[iv] = float(_DT)
        schema[bio] = "mass"
        schema[up] = "overwrite[map[float]]"
        schema[iv] = "overwrite[float]"
        state.update(
            crm_dfba_spec(
                _species_cfg(sp),
                dt=_DT,
                crm_name=f"crm_{name}",
                fba_name=f"fba_{name}",
                store_names={"biomass": bio, "uptakes": up, "interval": iv},
            )
        )
        emit_wires[bio] = [bio]

    state["emitter"] = emitter_from_wires(emit_wires)
    return {"schema": schema, "state": state}
