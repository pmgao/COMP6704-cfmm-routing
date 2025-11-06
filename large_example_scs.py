import json, numpy as np, cvxpy as cp

with open("huge128x4000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

n = data["n_tokens"]
m = data["m_pools"]
local_indices = data["local_indices"]
reserves_list = [np.array(r, dtype=float) for r in data["reserves"]]
fees = np.array(data["fees"], dtype=float)  # gamma = 1 - fee
pool_types = data["pool_types"]
weights_list = data["weights"]

# 1) build A_i (consistent with two-asset.py)
A = []
for l in local_indices:
    A_i = np.zeros((n, len(l)))
    for j, idx in enumerate(l):
        A_i[idx, j] = 1.0
    A.append(A_i)

# 2) Variables (same as in two-asset.py: Δ, Λ ≥ 0)
deltas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
lambdas = [cp.Variable(len(l), nonneg=True) for l in local_indices]

# 3) Network net trade vector Ψ
psi = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)])

# 4) This example does "liquidation": move token source -> target
src = data["experiment"]["source_token"]
tgt = data["experiment"]["target_token"]
t = 20000.0  # you can change to any value from data["experiment"]["amount_grid"]


current_assets = np.zeros(n)
current_assets[src] = t

# 5) Post-trade reserves R' = R + γΔ − Λ
new_R = [
    R + g * D - L for R, g, D, L in zip(reserves_list, fees, deltas, lambdas)
]

# 6) Constraints: product/weighted via log, sum is linear)
cons = [psi + current_assets >= 0]
log_eps = 1e-12
for k in range(m):
    typ = pool_types[k]
    cons.append(new_R[k] >= log_eps)
    if typ == "product":
        cons += [
            cp.sum(cp.log(new_R[k])) >= float(np.sum(np.log(reserves_list[k])))
        ]
    elif typ == "weighted":
        w = np.array(weights_list[k], dtype=float)
        cons += [w @ cp.log(new_R[k]) >= float(w @ np.log(reserves_list[k]))]
    elif typ == "sum":
        cons += [cp.sum(new_R[k]) >= cp.sum(reserves_list[k])]

# 7) Objective: maximize the amount of target token received (Ψ[tgt])
prob = cp.Problem(cp.Maximize(psi[tgt]), cons)

prob.solve(solver=cp.SCS, verbose=True)  # or CLARABEL/SCS/MOSEK
print("Max target received:", float(psi.value[tgt]))