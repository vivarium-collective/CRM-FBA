"""FBA as a process_bigraph Step.

The COBRApy FBA implementation now lives in the standalone **viva-cobra** library
(``viva_cobra.processes.fba``); this module re-exports it so crm_dfba keeps the
``crm_dfba.processes.fba.FBAStep`` address and API for back-compat while the logic
is single-sourced.

``FBAStep`` consumes CRM-predicted uptakes plus the current substrate/biomass
state, solves the FBA LP under uptake-derived exchange bounds, and returns deltas
for substrates and biomass. See viva-cobra for the full contract.
"""
from __future__ import annotations

# Single source of truth: viva-cobra. Re-exported for back-compat.
from viva_cobra.processes.fba import (  # noqa: F401
    FBAStep,
    _apply_bounds,
    _load_fba_model,
    apply_uptake_bounds,
)

__all__ = ["FBAStep", "apply_uptake_bounds", "_load_fba_model", "_apply_bounds"]
