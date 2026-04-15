from __future__ import annotations
from typing import Dict, Mapping
from crm_dfba.crms.base import BaseCRM


class MCRMCrm(BaseCRM):
    """
    Microbial Consumer Resource Model (MCRM) single-species uptake.

    In the full MCRM, per-capita growth is  mu * (sum_a C_a * W_a * R_a - T).
    The consumption term on resource a per unit biomass is C_a * R_a, which
    is what we use as the FBA exchange bound here.

    params:
        C: {resource: uptake_coefficient}
        W: {resource: energy_weight}      (optional, kept for provenance)
        T: maintenance (optional)
        mu: growth scaling (optional)
    """

    name = "mcrm"

    def _validate(self):
        C = self.params.get("C")
        if not isinstance(C, Mapping):
            raise ValueError("MCRMCrm requires params['C'] mapping resource -> coefficient")
        missing = set(self.resources) - set(C)
        if missing:
            raise ValueError(f"MCRMCrm missing C entries for: {sorted(missing)}")

    def compute_uptakes(
        self,
        resources: Mapping[str, float],
        biomass: float,
    ) -> Dict[str, float]:
        C = self.params["C"]
        out = {}
        for r in self.resources:
            R = max(0.0, float(resources.get(r, 0.0)))
            out[r] = float(C[r]) * R
        return out


class MiCRMCrm(BaseCRM):
    """
    Marsland/Goldford MiCRM single-species uptake.

    In the MiCRM, uptake of resource a is c_a * R_a (mass-action). Leakage
    and cross-feeding are handled by FBA stoichiometry, so here we just
    expose the uptake term.

    params:
        c: {resource: uptake_coefficient}
        leakage, w, m, g, rho, tau, D: optional, stored for provenance
    """

    name = "micrm"

    def _validate(self):
        c = self.params.get("c")
        if not isinstance(c, Mapping):
            raise ValueError("MiCRMCrm requires params['c'] mapping resource -> coefficient")
        missing = set(self.resources) - set(c)
        if missing:
            raise ValueError(f"MiCRMCrm missing c entries for: {sorted(missing)}")

    def compute_uptakes(
        self,
        resources: Mapping[str, float],
        biomass: float,
    ) -> Dict[str, float]:
        c = self.params["c"]
        out = {}
        for r in self.resources:
            R = max(0.0, float(resources.get(r, 0.0)))
            out[r] = float(c[r]) * R
        return out
