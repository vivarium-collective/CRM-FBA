"""
CRM-constrained Dynamic FBA
===========================

A process_bigraph Process that runs FBA with exchange-reaction bounds
derived from a configurable Consumer Resource Model (CRM). The CRM type
is selected from a registry (MacArthur, Adaptive, MCRM, MiCRM, Monod)
and its parameters are declared in config.

Coupling strategy
-----------------
'uptake_bounds': for each resource, the CRM-computed uptake rate u_a
becomes the FBA exchange lower bound via
    rxn.lower_bound = -u_a
FBA then solves for μ and realized exchange fluxes under stoichiometry.
Realized fluxes (not the CRM prediction) drive extracellular updates.
"""

from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Dict

import cobra
from cobra.io import load_model
from process_bigraph import Process

from crm_dfba.crms.registry import get_crm


warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")


def _load_fba_model(model_file: str, bounds: dict):
    """Load SBML file, a filename relative to a models dir, or a named cobra model."""
    search_dirs = [
        Path(__file__).resolve().parent / ".." / "models",
    ]
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
    """
    Coupling: CRM uptake rates become FBA exchange lower bounds.

    uptakes[resource] is a non-negative mmol/gDW/hr. Exchange convention in
    COBRA is negative for uptake, so we set lower_bound = -u. We also raise
    upper_bound if needed so the bound pair is consistent.
    """
    for resource, rxn_id in substrate_rxns.items():
        u = max(0.0, float(uptakes.get(resource, 0.0)))
        rxn = model.reactions.get_by_id(rxn_id)
        lb = -u
        if rxn.upper_bound < lb:
            rxn.upper_bound = lb
        rxn.lower_bound = lb


class CRMDynamicFBA(Process):
    """
    CRM-coupled dynamic FBA.

    Config
    ------
    model_file: str
        SBML file or named cobra model ("textbook", etc.).
    substrate_update_reactions: {resource: EX_rxn_id}
        Maps CRM resource names to FBA exchange reactions. The set of keys
        defines the CRM's resource axis.
    bounds: {rxn_id: {"lower": .., "upper": ..}}
        Static model bounds applied on load.
    crm: {"type": str, "params": {...}}
        CRM selection. type ∈ CRM_REGISTRY (macarthur|adaptive|mcrm|micrm|monod).
        params are passed to the CRM class; see each CRM for required keys.
    biomass_reaction: optional str
        If set, used as the FBA objective.

    Ports
    -----
    inputs/outputs follow spatio-flux DynamicFBA:
        substrates: map[concentration]   (in)  /  map[count] deltas (out)
        biomass:    mass                 (in)  /  mass delta         (out)
    """

    config_schema = {
        "model_file": "string",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
        "crm": {
            "type": "string",
            "params": "schema",
        },
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

        crm_cfg = config.get("crm") or {}
        crm_type = crm_cfg.get("type")
        if not crm_type:
            raise ValueError("config['crm']['type'] is required")
        self.crm = get_crm(
            name=crm_type,
            resources=self.resources,
            params=crm_cfg.get("params", {}),
        )

    def inputs(self):
        return {
            "substrates": "map[concentration]",
            "biomass": "mass",
        }

    def outputs(self):
        return {
            "substrates": "map[count]",
            "biomass": "mass",
        }

    def update(self, inputs, interval):
        substrates = inputs["substrates"]
        biomass = float(inputs["biomass"])

        uptakes = self.crm.compute_uptakes(substrates, biomass)
        apply_uptake_bounds(self.model, uptakes, self.substrate_rxns)

        solution = self.model.optimize()

        delta_subs = {r: 0.0 for r in self.resources}
        delta_biomass = 0.0

        if solution.status == "optimal":
            mu = float(solution.objective_value)
            delta_biomass = mu * biomass * interval

            for resource, rxn_id in self.substrate_rxns.items():
                flux = float(solution.fluxes[rxn_id]) * biomass * interval
                available = float(substrates.get(resource, 0.0))
                delta_subs[resource] = max(flux, -available)

        self.crm.step_internal_state(uptakes, substrates, biomass, interval)

        return {
            "substrates": delta_subs,
            "biomass": delta_biomass,
        }
