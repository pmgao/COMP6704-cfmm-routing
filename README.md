## Build Environment

```bash
sudo apt install python3-virtualenv
virtualenv venv
source ./venv/bin/activate

pip3 install cvxpy,numpy,matplotlib
pip3 install "cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,SCS,CLARABEL,QOCO,ECOS]"
```

## Run

The generate_cfmm_dateset.py generates the date according to input parameters, e.g,

```bash
python3 generate_cfmm_dataset.py \
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
  --out huge128x4000.json
```

And, the large_example.py can solve convex optimazation problem based on different input file and solver, which can be modified in code (just run python3 large_example.py simply). The plot_iter_xx.py and compare.py files can generate figures showing the solution status versus iteration and a comparison of different solvers, respectively (just run python3 xx.py simply).

The codebase is implemented by Wenxing Duan, Peimin Gao, Hiu Long Lee and Yulin Zhou.