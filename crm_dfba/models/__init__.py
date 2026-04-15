"""
Bundled genome-scale metabolic models (SBML) with friendly-name metadata.

Each entry declares the SBML filename (resolved relative to this directory),
the biomass objective, a canonical mapping of resource names → exchange
reactions, and the organism the model represents. CRMDynamicFBA resolves
``model_file`` directly against this directory, so configs can just say
``"model_file": "iAF1260.xml"`` (or ``"textbook"`` for the cobrapy-bundled
ecoli core).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

MODELS_DIR = Path(__file__).resolve().parent

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ecoli_core": {
        "model_file": "textbook",
        "organism": "Escherichia coli (core GSM, 95 rxns)",
        "biomass_reaction": "Biomass_Ecoli_core",
        "substrate_update_reactions": {
            "glucose": "EX_glc__D_e",
            "acetate": "EX_ac_e",
        },
        "default_bounds": {
            "EX_o2_e": {"lower": -20.0, "upper": 1000.0},
            "ATPM": {"lower": 1.0, "upper": 1.0},
        },
    },
    "iAF1260": {
        "model_file": "iAF1260.xml",
        "organism": "Escherichia coli (iAF1260, Feist et al. 2007)",
        "biomass_reaction": "BIOMASS_Ec_iAF1260_core_59p81M",
        "substrate_update_reactions": {
            "glucose": "EX_glc__D_e",
            "acetate": "EX_ac_e",
        },
        "default_bounds": {},
    },
    "iMM904": {
        "model_file": "iMM904.xml",
        "organism": "Saccharomyces cerevisiae (iMM904, Mo et al. 2009)",
        "biomass_reaction": "BIOMASS_SC5_notrace",
        "substrate_update_reactions": {
            "glucose": "EX_glc__D_e",
            "ammonium": "EX_nh4_e",
        },
        "default_bounds": {},
    },
    "iCN900": {
        "model_file": "iCN900.xml",
        "organism": "Clostridioides difficile (iCN900, Dannheim et al. 2017)",
        "biomass_reaction": "BIOMASS__5",
        "substrate_update_reactions": {
            "glucose": "EX_glc__D_e",
            "acetate": "EX_ac_e",
        },
        "default_bounds": {},
    },
    "iJN746": {
        "model_file": "iJN746.xml",
        "organism": "Pseudomonas putida (iJN746, Nogales et al. 2008)",
        "biomass_reaction": "BIOMASS_KT2440_WT3",
        "substrate_update_reactions": {
            "glucose": "EX_glc__D_e",
            "ammonium": "EX_nh4_e",
        },
        "default_bounds": {},
    },
    "iNF517": {
        "model_file": "iNF517.xml",
        "organism": "Lactococcus lactis (iNF517, Flahaut et al. 2013)",
        "biomass_reaction": "BIOMASS_LLA",
        "substrate_update_reactions": {
            "glucose": "EX_glc__D_e",
            "glutamate": "EX_glu__L_e",
        },
        "default_bounds": {},
    },
}


def get_model_spec(name: str) -> Dict[str, Any]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
