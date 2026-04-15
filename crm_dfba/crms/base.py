from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Mapping


class BaseCRM(ABC):
    """
    Consumer Resource Model base class.

    A CRM computes per-resource uptake rates (mmol / gDW / hr, positive = uptake)
    for a single species given current external resource concentrations and
    biomass. These uptake rates become the FBA exchange lower bounds via
    rxn.lower_bound = -uptake_rate in the coupled CRMDynamicFBA process.
    """

    name: str = "base"

    def __init__(self, resources, params):
        self.resources = list(resources)
        self.params = dict(params or {})
        self._validate()

    def _validate(self) -> None:
        pass

    @abstractmethod
    def compute_uptakes(
        self,
        resources: Mapping[str, float],
        biomass: float,
    ) -> Dict[str, float]:
        """Return {resource_name: uptake_rate_mmol_per_gDW_per_hr}."""
        ...

    def step_internal_state(self, uptakes, resources, biomass, interval):
        """Override for CRMs with internal state (e.g. Adaptive budget A)."""
        return None
