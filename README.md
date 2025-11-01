# CFMM Routing Experiments

> Reimplementation assets for the COMP6704 project exploring optimal routing across networks of constant function market makers.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Workflow](#workflow)
    * [1. Generate Synthetic CFMM Networks](#1-generate-synthetic-cfmm-networks)
    * [2. Solve a Routing Problem](#2-solve-a-routing-problem)
    * [3. Analyse Solver Iterations](#3-analyse-solver-iterations)
    * [4. Compare Solver Runtimes](#4-compare-solver-runtimes)
6. [Datasets](#datasets)
7. [Results Snapshot](#results-snapshot)
8. [Reproducing the Benchmarks](#reproducing-the-benchmarks)
9. [Troubleshooting & Tips](#troubleshooting--tips)
10. [Credits](#credits)

## Project Overview
This repository gathers the scripts, datasets, and plotting utilities that accompany our investigation of multi-pool routing in constant function market maker (CFMM) networks. We generate realistic synthetic liquidity configurations, formulate the routing optimisation problem in CVXPY, and evaluate a selection of convex solvers. The codebase is a cleaned-up derivative of the tooling originally authored by Wenxing Duan, Peimin Gao, Hiu Long Lee, and Yulin Zhou for COMP6704.

## Key Features
- **Dataset generator** that emits JSON (and optional NumPy) artefacts compatible with the [angeris/cfmm-routing-code](https://github.com/angeris/cfmm-routing-code) specification.
- **End-to-end optimisation example** showcasing how to load a dataset, assemble the convex programme, and solve a liquidation task.
- **Solver diagnostics tooling** for visualising iteration logs from Clarabel, ECOS, SCS, and MOSEK across multiple problem sizes.
- **Runtime comparison scripts** reproducing the summary plots used in our final report.

## Repository Structure
```
.
├── compare.py                # Generates the solver runtime comparison plot.
├── generate_cfmm_dataset.py  # CLI for synthetic CFMM network generation.
├── large_example.py          # Liquidation-style routing example solved with CVXPY.
├── plot_iter_cla.py          # Clarabel convergence visualisation.
├── plot_iter_ecos.py         # ECOS convergence visualisation.
├── plot_iter_mosek.py        # MOSEK convergence visualisation.
├── plot_iter_scs.py          # SCS convergence visualisation.
├── input/                    # Bundled benchmark datasets (JSON + optional NPZ).
└── output/                   # Cached plots created by the visualisation scripts.
```

## Environment Setup
We recommend Python ≥ 3.9 with an isolated virtual environment:

```bash
sudo apt update && sudo apt install python3-virtualenv
virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib cvxpy
# Install the solver back-ends you have licences for
pip install "cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,SCS,CLARABEL,QOCO,ECOS]"
```

> **Heads up:** commercial solvers such as MOSEK or Gurobi require separate licence activation before use.

## Workflow
The scripts are intended to be run independently so you can mix and match them in your own experiments.

### 1. Generate Synthetic CFMM Networks
`generate_cfmm_dataset.py` exposes a CLI for sampling heterogeneous pool networks. Key options include:

- `--n-tokens` / `--m-pools` for network scale.
- `--ratio-*` to control the mixture of constant-product, weighted, and constant-sum pools (values are renormalised to a simplex).
- `--pair-prob`, `--max-arity-nonweighted`, `--weighted-arities` to shape pool arity distributions.
- `--reserve-*` flags for reserve depth and dispersion via lognormal sampling and global scaling.
- `--fee-*` flags to choose between ranges or discrete basis-point tiers.
- `--connectivity` helpers (`none`, `backbone`, or `ring`) that add guaranteed constant-product edges.
- Experiment hints (`--task`, `--source-token`, `--target-token`, `--amount-grid`) embedded in the exported JSON.

Example command for the largest benchmark used in our study:

```bash
python generate_cfmm_dataset.py \
  --n-tokens 128 --m-pools 4000 \
  --ratio-product 0.35 --ratio-weighted 0.55 --ratio-sum 0.10 \
  --pair-prob 0.75 --weighted-arities 3 4 5 \
  --reserve-scale-min 2e4 --reserve-scale-max 5e6 \
  --reserve-log-mean 5.2 --reserve-log-sigma 1.1 \
  --fee-tiers-bps 5 30 50 \
  --connectivity ring --extra-backbone-weight 1 \
  --seed 3407 \
  --task purchase --source-token 0 --target-token 5 \
  --amount-grid 0 200000 41 \
  --out input/huge128x4000.json
```

Add `--save-npz` if you prefer loading arrays from a compressed NumPy archive.

### 2. Solve a Routing Problem
`large_example.py` demonstrates how to set up the liquidation problem discussed in class:

1. Load a dataset (default: `huge128x4000.json`) and build the selection matrices \(A_i\).
2. Declare non-negative decision variables `Δ` and `Λ` for each pool and aggregate the net trade vector \(\Psi\).
3. Inject your desired trade size via `current_assets` (the default sells 20,000 units of the source token).
4. Construct post-trade reserves \(R' = R + \gamma Δ - Λ\) and apply pool-specific convex constraints:
   - Constant-product pools enforce product preservation through log-sum constraints.
   - Weighted pools apply a weighted geometric mean inequality.
   - Constant-sum pools preserve the reserve sum.
5. Maximise the target token received and solve with MOSEK (default). Swap the solver for ECOS, SCS, or Clarabel by editing the `prob.solve(...)` call. A commented scaling snippet is included for first-order solvers.

Run the example from the repository root:

```bash
python large_example.py
```

### 3. Analyse Solver Iterations
The plotting scripts (`plot_iter_cla.py`, `plot_iter_ecos.py`, `plot_iter_scs.py`, `plot_iter_mosek.py`) parse stored console logs and emit publication-ready figures showing cost and residual trajectories. Each script saves a PDF alongside an interactive window if your environment supports it:

```bash
python plot_iter_ecos.py
```

The generated PDFs are stored under `output/` for convenience.

### 4. Compare Solver Runtimes
`compare.py` encodes the hand-recorded runtimes for Clarabel, ECOS, SCS, and MOSEK across six problem scales (from `8×12` up to `128×4000`). Running the script produces the `solver_runtime_vs_size_0.pdf` figure and prints the output path.

```bash
python compare.py
```

The inset plot highlights differences in the smaller regimes, while missing datapoints (e.g. Clarabel on large instances) are omitted automatically.

## Datasets
The `input/` directory ships with the JSON datasets used for our benchmarking campaign:

| File | Tokens × Pools | Notes |
| --- | --- | --- |
| `huge8x12.json` | 8 × 12 | Small sanity-check instance. |
| `huge32x100.json` | 32 × 100 | Medium network with mixed pool types. |
| `huge32x200.json` | 32 × 200 | Stress test for ECOS/SCS scaling. |
| `huge64x800.json` | 64 × 800 | Large sparse network. |
| `huge128x800.json` | 128 × 800 | Intermediate large-scale case. |
| `huge128x4000.json` | 128 × 4000 | Largest benchmark; used in `large_example.py`. |

Each JSON record matches the schema required by the optimisation scripts. If you pass `--save-npz` to the generator, the corresponding `.npz` file will appear beside the JSON for faster NumPy loading.

## Results Snapshot
The runtime comparison summarised in `compare.py` highlights the trade-offs between solvers:

| Solver | 8×12 | 32×100 | 32×200 | 64×800 | 128×800 | 128×4000 |
| --- | --- | --- | --- | --- | --- | --- |
| Clarabel | 0.003 s | 0.041 s | 0.063 s | — | — | — |
| SCS | 0.023 s | 0.616 s | 1.070 s | 38.900 s | 42.440 s | 212.190 s |
| ECOS | 0.002 s | 0.021 s | 0.036 s | 0.298 s | 1.451 s | 2.130 s |
| MOSEK | 0.013 s | 0.029 s | 0.045 s | 0.168 s | 0.188 s | 1.895 s |

Clarabel stalled on the two largest instances in our setup (denoted by `—`). ECOS delivered competitive performance across all sizes, while MOSEK provided the most consistent runtimes on the largest problems.

## Reproducing the Benchmarks
Follow the steps below to recreate the figures and optimisation results:

1. **Prepare the environment** using the instructions in [Environment Setup](#environment-setup). Install only the solvers you plan to run.
2. **Generate datasets** if you want to explore alternative configurations. Otherwise reuse the JSON files shipped in `input/`.
3. **Solve the routing problem** with `python large_example.py` and optionally tweak the trade size or solver choice.
4. **Produce iteration plots** by running any of the `plot_iter_*.py` scripts relevant to your solver logs.
5. **Regenerate the runtime figure** with `python compare.py`. The output PDF is written to `output/` (and the current directory).

## Troubleshooting & Tips
- Always fix the `--seed` argument when synthesising new datasets so that experiments remain reproducible.
- Large-scale problems solved with first-order methods (ECOS/SCS) benefit from the reserve scaling snippet included in `large_example.py`.
- MOSEK and other commercial solvers require licence activation before first use; check the solver documentation if you encounter authorisation errors.
- When comparing solvers, record both wall-clock time and iteration counts to disentangle convergence speed from per-iteration cost.

## Credits
This codebase adapts coursework originally produced by Wenxing Duan, Peimin Gao, Hiu Long Lee, and Yulin Zhou.
