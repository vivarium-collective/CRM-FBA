"""
Consumer Resource Model as a process_bigraph Process.

Reads current substrates and biomass; emits per-resource uptake rates
(mmol / gDW / hr, positive = uptake) that a downstream FBA step can
convert to exchange-reaction bounds. Also advances any CRM-internal
state (e.g. the adaptive allocation vector A in AdaptiveCRM).

The `interval` over which the CRM is advanced is also emitted, so a
non-temporal downstream Step can use it to convert fluxes into deltas.
"""
from __future__ import annotations
from typing import Dict

from process_bigraph import Process

from crm_dfba.crms.registry import get_crm


class CRMProcess(Process):
    config_schema = {
        "resources": "list[string]",
        "crm": {
            "type": "string",
            "params": "schema",
        },
    }

    def initialize(self, config):
        self.resources = list(config["resources"])
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
            "uptakes": "overwrite[map[float]]",
            "interval": "overwrite[float]",
        }

    def update(self, inputs, interval):
        substrates: Dict[str, float] = inputs["substrates"]
        biomass = float(inputs["biomass"])

        uptakes = self.crm.compute_uptakes(substrates, biomass)
        self.crm.step_internal_state(uptakes, substrates, biomass, interval)

        return {
            "uptakes": {r: float(uptakes.get(r, 0.0)) for r in self.resources},
            "interval": float(interval),
        }
