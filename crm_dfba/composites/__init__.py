"""Composite-generator registrations for the CRM-FBA composite.

Each generator is a thin ``@composite_generator`` wrapper around
``crm_dfba.demo.build_document`` for one CRM variant on the ecoli-core GSM,
so the vivarium-workbench composite browser can list and drill into the
CRM-constrained dynamic-FBA composite. The decorator side effect registers
each into process-bigraph's generator registry, which ``discover_generators``
reads.

Every composite wires ``CRMProcess -> FBAStep`` over shared substrate /
biomass / uptake / interval stores. crm_dfba's domain types + process links
are declared as ``core_extensions`` (``crm_dfba.core.register_types`` /
``register_processes``) so the composite realizes on whatever core the
workbench builds it with.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from crm_dfba.core import register_types, register_processes
from crm_dfba.demo import CRM_CONFIGS, build_document
from crm_dfba.composites.community import community_document


_CORE_EXTENSIONS = [register_types, register_processes]

# Initial extracellular pool + inoculum shared by the showcase composites.
_INITIAL_SUBSTRATES = {"glucose": 11.1, "acetate": 0.0}
_INITIAL_BIOMASS = 0.01
_DT = 0.05

# One-line phenomenology per CRM layer (keys match crm_dfba.demo.CRM_CONFIGS).
_DESCRIPTIONS = {
    "monod": "Monod kinetic uptake constraining dFBA on ecoli-core.",
    "macarthur": "MacArthur consumer-resource uptake constraining dFBA on ecoli-core.",
    "mcrm": "MCRM (mass-action) uptake constraining dFBA on ecoli-core.",
    "micrm": "MiCRM uptake constraining dFBA on ecoli-core.",
    "adaptive": "Adaptive (enzyme-allocation) uptake constraining dFBA on ecoli-core.",
}


def _make_generator(variant: str, crm_cfg: dict):
    @composite_generator(
        name=f"crm_fba_{variant}",
        description=_DESCRIPTIONS.get(variant, f"CRM-FBA ({variant}) on ecoli-core."),
        default_n_steps=100,
        core_extensions=_CORE_EXTENSIONS,
    )
    def _generator(core=None, _cfg=crm_cfg):
        return build_document(
            _cfg,
            dt=_DT,
            initial_substrates=_INITIAL_SUBSTRATES,
            initial_biomass=_INITIAL_BIOMASS,
        )

    _generator.__name__ = f"crm_fba_{variant}"
    return _generator


# Fire the decorators — one composite per CRM variant.
_GENERATORS = {
    variant: _make_generator(variant, cfg) for variant, cfg in CRM_CONFIGS.items()
}


# The showcase composite: a two-species community (glucose + acetate
# specialists) sharing one pool, cross-feeding via acetate overflow.
@composite_generator(
    name="crm_fba_community",
    description=(
        "Two-species E. coli community (glucose + acetate specialists) sharing "
        "one glucose/acetate pool: niche differentiation by acetate cross-feeding."
    ),
    default_n_steps=360,
    core_extensions=_CORE_EXTENSIONS,
)
def crm_fba_community(core=None):
    return community_document()
