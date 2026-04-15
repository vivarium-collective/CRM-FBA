"""
CRM-constrained Dynamic FBA
===========================

Composite of two pieces:

  - CRMProcess (Process): reads substrates + biomass, emits per-resource
    uptake rates and the interval it was advanced over. Owns the CRM
    instance and its internal state.
  - FBAStep (Step): reads uptakes + substrates + biomass + interval, sets
    exchange bounds, solves FBA, emits substrate/biomass deltas.

Wired together into a composite that reproduces the behavior of the
original monolithic CRMDynamicFBA process. The old class is retained as
``CRMDynamicFBAMonolithic`` for equivalence testing.

Coupling strategy
-----------------
'uptake_bounds': for each resource, CRM uptake u_a becomes the FBA
exchange lower bound (rxn.lower_bound = -u_a). FBA solves for mu and
realized exchange fluxes; realized fluxes (not CRM predictions) drive
extracellular updates.
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Mapping, Optional

from process_bigraph import Process

from crm_dfba.crms.registry import get_crm
from crm_dfba.processes.crm import CRMProcess
from crm_dfba.processes.fba import (
    FBAStep,
    _load_fba_model,
    apply_uptake_bounds,
)


warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")


CRM_ADDRESS = "local:!crm_dfba.processes.crm.CRMProcess"
FBA_ADDRESS = "local:!crm_dfba.processes.fba.FBAStep"


def crm_dfba_spec(
    config: Mapping,
    dt: float,
    *,
    crm_name: str = "crm",
    fba_name: str = "fba",
    store_names: Optional[Mapping[str, str]] = None,
) -> Dict[str, dict]:
    """
    Build the state fragment for a CRM-FBA composite.

    Parameters
    ----------
    config : same dict accepted by the old monolithic CRMDynamicFBA
        (``model_file``, ``substrate_update_reactions``, ``bounds``,
        ``biomass_reaction``, ``crm``).
    dt : interval at which the CRMProcess is scheduled.
    crm_name, fba_name : keys under which the two pieces appear in state.
    store_names : optional remapping of the four shared stores; keys are
        ``substrates``, ``biomass``, ``uptakes``, ``interval``.

    Returns
    -------
    A dict of state entries with two process/step specs. The caller is
    responsible for placing the shared stores (``substrates``,
    ``biomass``, ``uptakes``, ``interval``) and any emitter into the
    surrounding composite state.
    """
    names = {
        "substrates": "substrates",
        "biomass": "biomass",
        "uptakes": "uptakes",
        "interval": "interval",
    }
    if store_names:
        names.update(store_names)

    substrate_rxns: Dict[str, str] = dict(config["substrate_update_reactions"])
    resources: List[str] = list(substrate_rxns.keys())

    crm_spec = {
        "_type": "process",
        "address": CRM_ADDRESS,
        "config": {
            "resources": resources,
            "crm": dict(config["crm"]),
        },
        "interval": float(dt),
        "inputs": {
            "substrates": [names["substrates"]],
            "biomass": [names["biomass"]],
        },
        "outputs": {
            "uptakes": [names["uptakes"]],
            "interval": [names["interval"]],
        },
    }

    fba_spec = {
        "_type": "step",
        "address": FBA_ADDRESS,
        "config": {
            "model_file": config["model_file"],
            "substrate_update_reactions": substrate_rxns,
            "bounds": dict(config.get("bounds") or {}),
            "biomass_reaction": config.get("biomass_reaction"),
        },
        "inputs": {
            "uptakes": [names["uptakes"]],
            "substrates": [names["substrates"]],
            "biomass": [names["biomass"]],
            "interval": [names["interval"]],
        },
        "outputs": {
            "substrates": [names["substrates"]],
            "biomass": [names["biomass"]],
        },
    }

    return {crm_name: crm_spec, fba_name: fba_spec}


class CRMDynamicFBAMonolithic(Process):
    """
    Original single-process CRM-coupled dynamic FBA.

    Retained so the composite can be validated against it. For new usage
    prefer ``crm_dfba_spec`` + ``process_bigraph.Composite``.
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


CRMDynamicFBA = CRMDynamicFBAMonolithic
