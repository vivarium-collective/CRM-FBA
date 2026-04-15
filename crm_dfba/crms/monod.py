from __future__ import annotations
from typing import Dict, Mapping
from crm_dfba.crms.base import BaseCRM


class MonodCRM(BaseCRM):
    """
    Michaelis-Menten / Monod uptake. Matches the spatio-flux DynamicFBA kinetics
    so CRMDynamicFBA is a drop-in when crm.type == "monod".

    params:
        kinetic_params: {resource: (Km, Vmax)}
    """

    name = "monod"

    def _validate(self):
        kp = self.params.get("kinetic_params", {})
        missing = set(self.resources) - set(kp)
        if missing:
            raise ValueError(f"MonodCRM missing kinetic_params for: {sorted(missing)}")

    def compute_uptakes(
        self,
        resources: Mapping[str, float],
        biomass: float,
    ) -> Dict[str, float]:
        kp = self.params["kinetic_params"]
        out = {}
        for r in self.resources:
            Km, Vmax = kp[r]
            S = max(0.0, float(resources.get(r, 0.0)))
            out[r] = float(Vmax) * S / (float(Km) + S + 1e-12)
        return out
