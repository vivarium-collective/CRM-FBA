from __future__ import annotations
from typing import Dict, Type
from crm_dfba.crms.base import BaseCRM
from crm_dfba.crms.monod import MonodCRM
from crm_dfba.crms.macarthur import MacArthurCRM
from crm_dfba.crms.mcrm import MCRMCrm, MiCRMCrm
from crm_dfba.crms.adaptive import AdaptiveCRM


CRM_REGISTRY: Dict[str, Type[BaseCRM]] = {
    MonodCRM.name: MonodCRM,
    MacArthurCRM.name: MacArthurCRM,
    MCRMCrm.name: MCRMCrm,
    MiCRMCrm.name: MiCRMCrm,
    AdaptiveCRM.name: AdaptiveCRM,
}


def register_crm(cls: Type[BaseCRM]) -> Type[BaseCRM]:
    if not issubclass(cls, BaseCRM):
        raise TypeError(f"{cls} must subclass BaseCRM")
    CRM_REGISTRY[cls.name] = cls
    return cls


def get_crm(name: str, resources, params) -> BaseCRM:
    if name not in CRM_REGISTRY:
        raise KeyError(f"Unknown CRM type {name!r}. Available: {sorted(CRM_REGISTRY)}")
    return CRM_REGISTRY[name](resources, params)
