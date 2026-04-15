from __future__ import annotations
from typing import Dict, Mapping
import numpy as np
from crm_dfba.crms.base import BaseCRM


class MacArthurCRM(BaseCRM):
    """
    Classical MacArthur CRM single-species uptake kinetics.

    The CRM's per-resource consumption term defines the FBA exchange lower
    bound. Three resource-dynamics flavors from the vivarium-collective/CRM
    repo are supported, each with its own consumption functional form:

      resource_mode:
        'logistic' : u_a = c_a * R_a                 (consumption ∝ R)
        'external' : u_a = c_a * R_a                 (consumption ∝ R)
        'tilman'   : u_a = c_a                       (constant-rate, R-independent)

    MacArthur's growth-rate formula itself (sum_a w_a c_a R_a - m) is *not*
    used here for biomass — FBA's biomass stoichiometry determines μ. The
    w, m, r, K, tau parameters are accepted for parity with the CRM repo
    so a CRM-only ODE baseline can be reconstructed outside the process.

    params:
        c: mapping {resource: uptake_coefficient} — required
        resource_mode: 'logistic' | 'external' | 'tilman' (default 'logistic')
        w, m, tau, r, K: optional, unused by uptake but kept for provenance
    """

    name = "macarthur"

    def _validate(self):
        c = self.params.get("c")
        if not isinstance(c, Mapping):
            raise ValueError("MacArthurCRM requires params['c'] mapping resource -> coefficient")
        missing = set(self.resources) - set(c)
        if missing:
            raise ValueError(f"MacArthurCRM missing c entries for: {sorted(missing)}")
        mode = self.params.get("resource_mode", "logistic")
        if mode not in ("logistic", "external", "tilman"):
            raise ValueError(f"Unknown resource_mode {mode!r}")

    def compute_uptakes(
        self,
        resources: Mapping[str, float],
        biomass: float,
    ) -> Dict[str, float]:
        c = self.params["c"]
        mode = self.params.get("resource_mode", "logistic")
        out = {}
        for r in self.resources:
            R = max(0.0, float(resources.get(r, 0.0)))
            if mode == "tilman":
                out[r] = float(c[r])
            else:
                out[r] = float(c[r]) * R
        return out
