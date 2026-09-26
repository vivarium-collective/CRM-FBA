"""Workspace core factory for the vivarium-workbench.

The workbench imports ``<package>.core.build_core`` to obtain a
process-bigraph core with this package's types + processes registered, so it
can introspect the Registry and realize composites on a core that knows
crm_dfba's domain types and links.

crm_dfba's schemas use a handful of domain type-names (``concentration``,
``mass``, ``count``, ``bounds``) that are not part of the process-bigraph
base types. They are registered here (and re-applied as composite
``core_extensions``) so composites realize on whatever core builds them.
Process specs address their classes by dynamic-import address
(``local:!crm_dfba.processes.crm.CRMProcess`` etc.); the link registrations
below are for discovery + friendly names in the workbench.
"""
from __future__ import annotations

from process_bigraph import allocate_core

from crm_dfba.processes.crm import CRMProcess
from crm_dfba.processes.fba import FBAStep
from crm_dfba.processes.crm_dfba import (
    CRMDynamicFBA,
    CRMDynamicFBAMonolithic,
)


# Domain types used across crm_dfba's process interfaces + composite schemas.
# concentration / mass / count are non-negative scalars (floats); bounds is a
# per-reaction {lower, upper} exchange-flux window (map[bounds] in configs).
_TYPE_DEFS = {
    "concentration": {"_type": "float", "_default": 0.0},
    "mass": {"_type": "float", "_default": 0.0},
    "count": {"_type": "float", "_default": 0.0},
    "bounds": {
        "lower": {"_type": "float", "_default": 0.0},
        "upper": {"_type": "float", "_default": 1000.0},
    },
}


def _is_registered(core, name: str) -> bool:
    """True if ``name`` resolves to a real schema (unknown names echo back)."""
    try:
        resolved = core.access(name)
    except Exception:
        return False
    return not (isinstance(resolved, str) and resolved == name)


def register_types(core):
    """Register crm_dfba's domain types on ``core`` (idempotent)."""
    for name, schema in _TYPE_DEFS.items():
        if not _is_registered(core, name):
            core.register_type(name, schema)
    return core


def register_processes(core):
    """Register crm_dfba's process/step links under friendly names."""
    core.register_link("CRMProcess", CRMProcess)
    core.register_link("FBAStep", FBAStep)
    core.register_link("CRMDynamicFBA", CRMDynamicFBA)
    core.register_link("CRMDynamicFBAMonolithic", CRMDynamicFBAMonolithic)
    return core


def build_core(core=None):
    """Return a process-bigraph core with crm_dfba's types + processes.

    Pass an existing ``core`` to compose crm_dfba's registrations onto it
    (so a downstream repo's ``build_core`` can inherit them); omit it to get
    a fresh ``allocate_core()``.
    """
    if core is None:
        core = allocate_core()
    register_types(core)
    register_processes(core)
    return core
