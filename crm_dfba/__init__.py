from crm_dfba.processes.crm_dfba import (
    CRMDynamicFBA,
    CRMDynamicFBAMonolithic,
    crm_dfba_spec,
)
from crm_dfba.processes.crm import CRMProcess
from crm_dfba.processes.fba import FBAStep
from crm_dfba.crms.registry import CRM_REGISTRY, get_crm, register_crm
from crm_dfba.models import MODEL_REGISTRY, get_model_spec

__all__ = [
    "CRMDynamicFBA",
    "CRMDynamicFBAMonolithic",
    "CRMProcess",
    "FBAStep",
    "crm_dfba_spec",
    "CRM_REGISTRY", "get_crm", "register_crm",
    "MODEL_REGISTRY", "get_model_spec",
]

# Fire the @composite_generator side-effects so the vivarium-workbench composite
# browser discovers the CRM-FBA composites. Imported last, after the names above
# are bound (crm_dfba.composites imports crm_dfba.demo, which imports
# crm_dfba_spec from this module).
from crm_dfba import composites as _composites  # noqa: E402,F401
