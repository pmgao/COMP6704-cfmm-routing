#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, numpy as np
from pathlib import Path
from typing import List

def make_parser():
    # Build an argument parser for generating synthetic CFMM routing datasets
    p = argparse.ArgumentParser(
        description="Generate synthetic CFMM routing datasets compatible with angeris/cfmm-routing-code."
    )
    # scale / size
    p.add_argument("--n-tokens", type=int, default=8, help="Number of global tokens (assets).")
    p.add_argument("--m-pools", type=int, default=60, help="Number of CFMM pools to create.")
    # composition of pool types
    p.add_argument("--ratio-product", type=float, default=0.55, help="Fraction of constant-product pools.")
    p.add_argument("--ratio-weighted", type=float, default=0.30, help="Fraction of weighted geometric-mean pools.")
    p.add_argument("--ratio-sum", type=float, default=0.15, help="Fraction of constant-sum pools.")
    # arities
    p.add_argument("--pair-prob", type=float, default=0.8, help="Probability that non-weighted pools are 2-asset (else 3-asset).")
    p.add_argument("--max-arity-nonweighted", type=int, default=3, help="Max arity for product/sum pools (2 or 3).")
    p.add_argument("--weighted-arities", type=int, nargs="+", default=[3,4], help="Allowed arities for weighted pools.")
    # reserves (depth)
    p.add_argument("--reserve-scale-min", type=float, default=5e2, help="Min scale factor to control pool depth.")
    p.add_argument("--reserve-scale-max", type=float, default=5e4, help="Max scale factor to control pool depth.")
    p.add_argument("--reserve-log-mean", type=float, default=4.5, help="Lognormal mean for raw samples.")
    p.add_argument("--reserve-log-sigma", type=float, default=0.9, help="Lognormal sigma for raw samples.")
    # fees (gamma = 1 - fee)
    p.add_argument("--fee-bps-min", type=float, default=5.0, help="Min fee in basis points (bps), e.g., 5=0.05%%.")
    p.add_argument("--fee-bps-max", type=float, default=30.0, help="Max fee in bps, e.g., 30=0.30%%.")
    p.add_argument("--fee-tiers-bps", type=float, nargs="*", default=[], help="Optional discrete fee tiers in bps; if provided, sample from these.")
    # topology / connectivity helpers
    p.add_argument("--connectivity", choices=["none","backbone","ring"], default="backbone",
                   help="Add guaranteed connectivity edges among consecutive tokens.")
    p.add_argument("--extra-backbone-weight", type=int, default=1,
                   help="How many extra guaranteed product pools to add between consecutive tokens (for connectivity).")
    # random seed
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    # experiment hints
    p.add_argument("--task", choices=["liquidate","purchase","arbitrage"], default="liquidate",
                   help="Optional experiment hint.")
    p.add_argument("--source-token", type=int, default=0, help="Source token index for liquidate experiments.")
    p.add_argument("--target-token", type=int, default=-1, help="Target token index; default is last token.")
    p.add_argument("--amount-grid", type=float, nargs=3, default=[0.0, 50000.0, 61.0],
                   help="Grid for amounts: start stop num")
    # outputs
    p.add_argument("--out", type=str, default="cfmm_generated.json", help="Output JSON path.")
    p.add_argument("--save-npz", action="store_true", help="Also save an NPZ with same fields (ragged arrays as object).")
    return p

def rng_setup(seed:int):
    # Initialize a NumPy random number generator with a fixed seed
    return np.random.default_rng(seed)

def sample_gamma(rng, fee_bps_min: float, fee_bps_max: float, fee_tiers_bps: List[float]):
    # Sample trading fee (gamma = 1 - fee). If discrete tiers provided, sample from them; otherwise uniform in [min,max].
    if fee_tiers_bps:
        bps = float(rng.choice(np.array(fee_tiers_bps)))
    else:
        bps = float(rng.uniform(fee_bps_min, fee_bps_max))
    fee = bps / 1e4
    gamma = 1.0 - fee
    return float(gamma)

def sample_reserves(rng, k: int, log_mean: float, log_sigma: float,
                    scale_min: float, scale_max: float):
    # Sample k reserves using a lognormal distribution, then scale to a random depth and normalize around the mean
    raw = rng.lognormal(mean=log_mean, sigma=log_sigma, size=k)
    scale = float(rng.uniform(scale_min, scale_max))
    vals = np.maximum(1.0, raw * scale / np.mean(raw))
    return [float(v) for v in vals]

def sample_weights(rng, k: int):
    # Sample k positive weights biased away from zero, then normalize to sum to 1
    w = rng.random(k) + 0.2
    w = w / w.sum()
    return [float(x) for x in w]

def sample_token_subset(rng, k: int, n: int):
    # Sample k distinct token indices from 0..n-1 and return them sorted
    inds = rng.choice(np.arange(n), size=k, replace=False)
    return sorted([int(i) for i in inds])

def sample_nonweighted_k(rng, pair_prob: float, max_arity: int):
    # For non-weighted pools, choose arity 2 vs 3, respecting a max-arity cap
    if max_arity <= 2:
        return 2
    return int(rng.choice([2,3], p=[pair_prob, 1.0-pair_prob]))

def choose_pool_type(rng, ratios):
    # Choose a pool type according to provided ratios
    kinds = ["product","weighted","sum"]
    return str(rng.choice(kinds, p=ratios))

def build_dataset(
    n_tokens:int, m_pools:int,
    ratios,
    pair_prob:float, max_arity_nonweighted:int, weighted_arities,
    reserve_scale_min:float, reserve_scale_max:float, reserve_log_mean:float, reserve_log_sigma:float,
    fee_bps_min:float, fee_bps_max:float, fee_tiers_bps,
    connectivity:str, extra_backbone_weight:int,
    seed:int,
    task:str, source_token:int, target_token:int, amount_grid
):
    # Main generator that builds a synthetic dataset of pools with metadata and experiment hints
    rng = rng_setup(seed)
    pools = []

    # main random pools
    for _ in range(m_pools):
        # Sample pool type and its arity
        ptype = choose_pool_type(rng, ratios)
        if ptype == "weighted":
            k = int(rng.choice(np.array(weighted_arities)))
        else:
            k = sample_nonweighted_k(rng, pair_prob, max_arity_nonweighted)
        # Select tokens, fees (gamma), reserves, and weights (if weighted)
        tokens = sample_token_subset(rng, k, n_tokens)
        gamma = sample_gamma(rng, fee_bps_min, fee_bps_max, fee_tiers_bps)
        reserves = sample_reserves(rng, k, reserve_log_mean, reserve_log_sigma, reserve_scale_min, reserve_scale_max)
        weights = sample_weights(rng, k) if ptype == "weighted" else None
        pools.append({
            "type": ptype,
            "tokens": tokens,
            "reserves": reserves,
            "gamma": gamma,
            "weights": weights
        })

    # connectivity helpers
    # Optionally enforce a backbone or ring connectivity by adding extra 2-asset product pools between consecutive tokens
    if connectivity != "none":
        reps = max(1, int(extra_backbone_weight))
        seq = list(range(n_tokens))
        if connectivity == "ring" and n_tokens > 2:
            seq.append(0)
        for _ in range(reps):
            for a, b in zip(seq[:-1], seq[1:]):
                pools.append({
                    "type": "product",
                    "tokens": [int(a), int(b)],
                    "reserves": sample_reserves(rng, 2, reserve_log_mean, reserve_log_sigma, reserve_scale_min, reserve_scale_max),
                    "gamma": sample_gamma(rng, fee_bps_min, fee_bps_max, fee_tiers_bps),
                    "weights": None
                })

    # Collect fields for output structure
    global_indices = list(range(n_tokens))
    local_indices  = [p["tokens"] for p in pools]
    reserves       = [p["reserves"] for p in pools]
    fees           = [p["gamma"]   for p in pools]
    pool_types     = [p["type"]    for p in pools]
    weights_list   = [p["weights"] for p in pools]
    if target_token < 0:
        target_token = n_tokens - 1

    # Build dataset dictionary compatible with downstream tooling
    dataset = {
        "n_tokens": n_tokens,
        "m_pools": len(pools),
        "global_indices": global_indices,
        "local_indices": local_indices,
        "reserves": reserves,
        "fees": fees,
        "pool_types": pool_types,
        "weights": weights_list,
        "experiment": {
            "task": task,
            "source_token": int(source_token),
            "target_token": int(target_token),
            "amount_grid": {
                "start": float(amount_grid[0]),
                "stop": float(amount_grid[1]),
                "num": int(amount_grid[2])
            }
        }
    }
    return dataset

def save_dataset(dataset, out_path:str, save_npz:bool):
    # Save dataset to JSON and optionally to NPZ (with ragged arrays stored as dtype=object)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    if save_npz:
        npz_path = out.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            global_indices=np.array(dataset["global_indices"], dtype=int),
            local_indices=np.array(dataset["local_indices"], dtype=object),
            reserves=np.array(dataset["reserves"], dtype=object),
            fees=np.array(dataset["fees"], dtype=float),
            pool_types=np.array(dataset["pool_types"], dtype=object),
            weights=np.array(dataset["weights"], dtype=object)
        )
    return str(out)

def main_cli():
    # Parse CLI args, normalize ratios, build dataset, and save to disk
    args = make_parser().parse_args()
    ratios = np.array([args.ratio_product, args.ratio_weighted, args.ratio_sum], dtype=float)
    s = ratios.sum()
    if s <= 0:
        raise SystemExit("Pool-type ratios must sum to a positive number.")
    ratios = (ratios / s).tolist()
    ds = build_dataset(
        n_tokens=args.n_tokens, m_pools=args.m_pools,
        ratios=ratios,
        pair_prob=args.pair_prob, max_arity_nonweighted=args.max_arity_nonweighted, weighted_arities=args.weighted_arities,
        reserve_scale_min=args.reserve_scale_min, reserve_scale_max=args.reserve_scale_max,
        reserve_log_mean=args.reserve_log_mean, reserve_log_sigma=args.reserve_log_sigma,
        fee_bps_min=args.fee_bps_min, fee_bps_max=args.fee_bps_max, fee_tiers_bps=args.fee_tiers_bps,
        connectivity=args.connectivity, extra_backbone_weight=args.extra_backbone_weight,
        seed=args.seed,
        task=args.task, source_token=args.source_token, target_token=args.target_token,
        amount_grid=args.amount_grid
    )
    path = save_dataset(ds, args.out, args.save_npz)
    print(f"Saved dataset to: {path}")
    if args.save_npz:
        print(f"Also saved: {Path(path).with_suffix('.npz')}")

if __name__ == "__main__":
    main_cli()