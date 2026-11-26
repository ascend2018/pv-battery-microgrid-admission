# Hypervolume-prioritized admission for PV–battery microgrids

This repository accompanies *Grid-connected PV-battery microgrid energy management with hypervolume-prioritized admission*. It includes DA-HEMORL-HA, baseline components, and runnable examples.

DA-HEMORL-HA evaluates a parent policy and its SAC-refined candidate on a shared schedule. It uses feasibility, Pareto dominance, and hypervolume to decide which policies receive priority in the next population.

## Code overview

| Component | Source |
| --- | --- |
| DA-HEMORL-HA case entry and paired admission | `run_main_case.py`, `src/hybrid/` |
| Microgrid environment and battery wear | `src/env/`, `src/degradation/` |
| Neural NSGA-II and Neural NSGA-III population selection | `src/baselines/neural_nsga.py` |
| Adapted CE-CMEPS | `src/jeet_r2/ce.py` |
| Nominal FG-MPC and the fixed-SoC Rule | `src/jeet_r2/mpc.py`, `src/baselines/rule_based.py` |

## Run the component examples

Use Python 3.12 to run the examples for policy admission, population selection, and baseline behavior.

```sh
python -m pip install -r requirements.txt
python check_admission.py
python check_ce_comparator.py
python check_neural_baselines.py
python check_engineering_references.py
```

## Run the main method

With CUDA-enabled PyTorch, run the included example case using its illustrative 24-hour profiles.

```sh
python run_main_case.py examples/synthetic_case.json --output policies.json
```

The input JSON specifies a seed and an ordered schedule of 24- or 48-hour training units, with operating profiles and forecast windows for each unit. The output JSON records the last evaluated archive with policy parameters and objective values, and the last constructed population. Run `python run_main_case.py --help` for the input fields.

## Study data

The paper and supplementary material describe the simulation and evaluation settings. The corresponding author provides the study data upon reasonable request.
