"""Bundled genome-scale metabolic models (SBML) with friendly-name metadata.

The bundled GEMs and their registry now live in the standalone **viva-cobra**
library (``viva_cobra.models``); this module re-exports them so crm_dfba keeps the
``crm_dfba.models`` API for back-compat while the SBML files + registry are
single-sourced. ``FBAStep`` (also from viva-cobra) resolves ``model_file``
against ``viva_cobra/models/`` (or ``$VIVA_COBRA_MODELS_DIR``), so configs can
still say ``"model_file": "iAF1260.xml"`` or ``"textbook"``.
"""
from __future__ import annotations

# Single source of truth: viva-cobra. Re-exported for back-compat.
from viva_cobra.models import (  # noqa: F401
    MODEL_REGISTRY,
    MODELS_DIR,
    get_model_spec,
)

__all__ = ["MODEL_REGISTRY", "MODELS_DIR", "get_model_spec"]
