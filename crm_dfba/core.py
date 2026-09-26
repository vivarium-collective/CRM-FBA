"""Workspace core factory for the vivarium-workbench.

The workbench imports ``<package>.core.build_core`` to obtain a
process-bigraph core with this package's processes registered, so it can
introspect the Registry (Processes tab) and realize composites on a core
that knows crm_dfba's links.

crm_dfba's composite specs address their processes by dynamic-import
address (``local:!crm_dfba.processes.crm.CRMProcess`` etc.), so the
registrations below are for discovery + friendly names in the workbench,
not a hard requirement for running the composite standalone.
"""
from __future__ import annotations

from process_bigraph import allocate_core

from crm_dfba.processes.crm import CRMProcess
from crm_dfba.processes.fba import FBAStep
from crm_dfba.processes.crm_dfba import (
    CRMDynamicFBA,
    CRMDynamicFBAMonolithic,
)


def build_core(core=None):
    """Return a process-bigraph core with crm_dfba's processes registered.

    Pass an existing ``core`` to compose crm_dfba's registrations onto it
    (so a downstream repo's ``build_core`` can inherit them); omit it to get
    a fresh ``allocate_core()``.
    """
    if core is None:
        core = allocate_core()
    core.register_link("CRMProcess", CRMProcess)
    core.register_link("FBAStep", FBAStep)
    core.register_link("CRMDynamicFBA", CRMDynamicFBA)
    core.register_link("CRMDynamicFBAMonolithic", CRMDynamicFBAMonolithic)
    return core
