"""
FBA as a process_bigraph Step.

Non-temporal: consumes CRM-predicted uptakes plus the current substrate
and biomass state, solves the FBA LP under uptake-derived exchange
bounds, and returns deltas for substrates and biomass. The `interval`
over which those deltas apply is read from the input state (emitted by
the upstream CRM Process).

Coupling (same as the monolithic CRMDynamicFBA):
  uptake u_a (mmol/gDW/hr, >=0)  -> exchange lower_bound = -u_a
  realized flux v_a              -> delta_substrate = v_a * biomass * dt
  growth rate mu                 -> delta_biomass    = mu  * biomass * dt
Substrate deltas are clamped to available mass.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Dict

import cobra
from cobra.io import load_model
from process_bigraph import Step


warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")


def _load_fba_model(model_file: str, bounds: dict):
    search_dirs = [Path(__file__).resolve().parent / ".." / "models"]
    env_dir = os.environ.get("CRM_DFBA_MODELS_DIR")
    if env_dir:
        search_dirs.insert(0, Path(env_dir))

    if model_file.endswith(".xml") or model_file.endswith(".sbml"):
        for d in search_dirs:
            p = (d / model_file).resolve()
            if p.exists():
                return _apply_bounds(cobra.io.read_sbml_model(str(p)), bounds)
        p = Path(model_file)
        if p.exists():
            return _apply_bounds(cobra.io.read_sbml_model(str(p)), bounds)
        raise FileNotFoundError(f"SBML file not found: {model_file}")

    return _apply_bounds(load_model(model_file), bounds)


def _apply_bounds(model, bounds):
    for rxn_id, limits in (bounds or {}).items():
        rxn = model.reactions.get_by_id(rxn_id)
        lower = limits.get("lower")
        upper = limits.get("upper")
        if lower is not None:
            rxn.lower_bound = float(lower)
        if upper is not None:
            rxn.upper_bound = float(upper)
    return model


def apply_uptake_bounds(model, uptakes: Dict[str, float], substrate_rxns: Dict[str, str]) -> None:
    for resource, rxn_id in substrate_rxns.items():
        u = max(0.0, float(uptakes.get(resource, 0.0)))
        rxn = model.reactions.get_by_id(rxn_id)
        lb = -u
        if rxn.upper_bound < lb:
            rxn.upper_bound = lb
        rxn.lower_bound = lb


class FBAStep(Step):
    config_schema = {
        "model_file": "string",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
        "biomass_reaction": "maybe[string]",
    }

    def initialize(self, config):
        self.substrate_rxns: Dict[str, str] = dict(config["substrate_update_reactions"])
        self.resources = list(self.substrate_rxns.keys())

        self.model = _load_fba_model(
            model_file=config["model_file"],
            bounds=config.get("bounds", {}) or {},
        )
        biomass_rxn = config.get("biomass_reaction")
        if biomass_rxn:
            self.model.objective = biomass_rxn

    def inputs(self):
        return {
            "uptakes": "map[float]",
            "substrates": "map[concentration]",
            "biomass": "mass",
            "interval": "float",
        }

    def triggers(self):
        return {"uptakes": "map[float]"}

    def outputs(self):
        return {
            "substrates": "map[count]",
            "biomass": "mass",
        }

    def update(self, state):
        uptakes = state.get("uptakes", {}) or {}
        substrates = state.get("substrates", {}) or {}
        biomass = float(state.get("biomass", 0.0))
        interval = float(state.get("interval", 0.0))

        apply_uptake_bounds(self.model, uptakes, self.substrate_rxns)
        solution = self.model.optimize()

        delta_subs = {r: 0.0 for r in self.resources}
        delta_biomass = 0.0

        if solution.status == "optimal" and interval > 0.0:
            mu = float(solution.objective_value)
            delta_biomass = mu * biomass * interval

            for resource, rxn_id in self.substrate_rxns.items():
                flux = float(solution.fluxes[rxn_id]) * biomass * interval
                available = float(substrates.get(resource, 0.0))
                delta_subs[resource] = max(flux, -available)

        return {
            "substrates": delta_subs,
            "biomass": delta_biomass,
        }
