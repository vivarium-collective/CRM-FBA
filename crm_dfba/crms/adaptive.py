from __future__ import annotations
from typing import Dict, Mapping
import numpy as np
from crm_dfba.crms.base import BaseCRM


class AdaptiveCRM(BaseCRM):
    """
    Picciani-Mori adaptive CRM (single species variant).

    The species carries an internal allocation vector A_a >= 0 (one entry
    per resource) with an energy budget E_star. The uptake of resource a is

        u_a = A_a * v_a * R_a / (K_a + R_a)

    and A evolves between FBA steps according to

        dA_a/dt = A_a * ( lam * v_a * r_a - penalty )
        penalty = active * (sum_b A_b / E_star) * growth
        growth  = sum_a A_a * v_a * r_a
        active  = 1 if sum_b A_b >= E_star else 0

    Here the "growth" term used for the adaptation dynamics is the CRM-side
    growth proxy (sum_a A_a v_a r_a); FBA still determines the realized
    biomass μ from stoichiometry.

    params:
        v: {resource: v_a} max uptake rate per unit allocation
        K: {resource: K_a} half-saturation
        lam: float, adaptation speed
        E_star: float, allocation budget
        A0: {resource: A0_a}, initial allocation (default equal split of E_star)
    """

    name = "adaptive"

    def _validate(self):
        for key in ("v", "K"):
            m = self.params.get(key)
            if not isinstance(m, Mapping):
                raise ValueError(f"AdaptiveCRM requires params[{key!r}] mapping")
            missing = set(self.resources) - set(m)
            if missing:
                raise ValueError(f"AdaptiveCRM missing {key} for: {sorted(missing)}")
        for key in ("lam", "E_star"):
            if key not in self.params:
                raise ValueError(f"AdaptiveCRM requires params[{key!r}]")

        E_star = float(self.params["E_star"])
        n = len(self.resources)
        A0_map = self.params.get("A0")
        if A0_map is None:
            share = E_star / n if n else 0.0
            A0_map = {r: share for r in self.resources}
        self._A = np.array([float(A0_map[r]) for r in self.resources], dtype=float)

    def _r_frac(self, resources):
        K = self.params["K"]
        out = np.zeros(len(self.resources))
        for i, r in enumerate(self.resources):
            R = max(0.0, float(resources.get(r, 0.0)))
            out[i] = R / (float(K[r]) + R + 1e-12)
        return out

    def compute_uptakes(
        self,
        resources: Mapping[str, float],
        biomass: float,
    ) -> Dict[str, float]:
        v = self.params["v"]
        r_frac = self._r_frac(resources)
        v_vec = np.array([float(v[r]) for r in self.resources])
        u = self._A * v_vec * r_frac
        return {r: float(u[i]) for i, r in enumerate(self.resources)}

    def step_internal_state(self, uptakes, resources, biomass, interval):
        """
        Substep the adaptation ODE inside the outer FBA interval so the
        dynamics remain stable at moderate lam. After each substep we soft-
        cap sum(A) at E_star by rescaling; this matches the intent of the
        penalty term (push budget toward E_star) without the overshoot that
        makes an explicit-Euler step at the outer dt diverge.
        """
        v = self.params["v"]
        lam = float(self.params["lam"])
        E_star = float(self.params["E_star"])
        n_sub = int(self.params.get("n_substeps", 50))

        v_vec = np.array([float(v[r]) for r in self.resources])
        r_frac = self._r_frac(resources)   # resources are held constant across substeps
        sub_dt = float(interval) / max(n_sub, 1)

        A = self._A.copy()
        for _ in range(n_sub):
            growth = float(np.sum(A * v_vec * r_frac))
            budget = float(np.sum(A))
            active = 1.0 if budget >= E_star else 0.0
            penalty = active * (budget / max(E_star, 1e-12)) * growth
            dA = A * (lam * v_vec * r_frac - penalty)
            A = A + dA * sub_dt
            A = np.maximum(A, 0.0)
            tot = A.sum()
            if tot > E_star:   # hard budget cap for numerical stability
                A *= E_star / tot
        self._A = A
