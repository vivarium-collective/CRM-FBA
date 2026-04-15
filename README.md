# CRM-FBA

### **[→ Experiment report](https://vivarium-collective.github.io/CRM-FBA/)**

[![community experiment preview](docs/community.png)](https://vivarium-collective.github.io/CRM-FBA/)

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
**composite** that runs dynamic Flux Balance Analysis with exchange bounds
derived from a configurable **Consumer Resource Model**. Two pieces wired
together over shared stores:

- `CRMProcess` (Process) — owns the CRM instance, reads substrates +
  biomass, emits per-resource uptake rates and the interval it was
  advanced over.
- `FBAStep` (Step) — reads those uptakes, sets exchange lower bounds on
  a COBRA model, solves FBA, and emits realized substrate/biomass deltas.

The CRM layer is pluggable via a registry (MacArthur, Adaptive, MCRM,
MiCRM, Monod). A monolithic equivalent (`CRMDynamicFBAMonolithic`) is
kept for equivalence testing.

## Install

```bash
pip install -e .
```

## Run the experiment suite

```bash
python -m crm_dfba.experiments.test_suite
```

Runs seven experiments on bundled genome-scale metabolic models and writes
`docs/index.html`. See the [live report](https://vivarium-collective.github.io/CRM-FBA/)
for what's inside.

## Use the composite

`CRMDynamicFBA` is a thin wrapper that builds the CRMProcess + FBAStep
composite from a single config dict:

```python
from crm_dfba import CRMDynamicFBA

proc = CRMDynamicFBA(config={
    "model_file": "textbook",                  # or iAF1260.xml, iMM904.xml, ...
    "biomass_reaction": "Biomass_Ecoli_core",
    "substrate_update_reactions": {
        "glucose": "EX_glc__D_e",
        "acetate": "EX_ac_e",
    },
    "bounds": {"EX_o2_e": {"lower": -20, "upper": None}},
    "crm": {
        "type": "macarthur",                   # monod | macarthur | mcrm | micrm | adaptive
        "params": {"c": {"glucose": 0.9, "acetate": 0.2},
                   "resource_mode": "external"},
    },
}, core=core)
```

Each CRM declares its own `params` shape — see `crm_dfba/crms/*.py`.
Register a new CRM with `register_crm(MyCRM)` (subclass `BaseCRM`).

To wire the pieces directly (e.g. embedding in a larger bigraph), use
`crm_dfba_spec(config, dt)` to get the two-node state fragment, or
instantiate `CRMProcess` and `FBAStep` yourself.

## Bundled GSMs

Six SBML models under `crm_dfba/models/` with a friendly-name registry
(`crm_dfba.models.MODEL_REGISTRY`):
`ecoli_core` (textbook), `iAF1260` (E. coli), `iMM904` (S. cerevisiae),
`iCN900` (C. difficile), `iJN746` (P. putida), `iNF517` (L. lactis).
