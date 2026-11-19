#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compute_node_features.py - thresholded nodal feature cache.

For each participant and resolution the connectivity matrix is thresholded and six nodal
measures are computed.
  1. Clustering: bct.clustering_coef_wu assumes weights in [0, 1] but does not normalize
     them, so the clustering coefficient is computed after dividing each participant's
     thresholded matrix by its own maximum weight (same operation as
     bct.weight_conversion(W, 'normalize')). The other five measures use raw weights.
  2. A density and phantom-edge report is saved to edge_report_<res>.json (per participant:
     actual density, target edge count, phantom edges).
  3. A missing matrix or multiple matches raise an explicit error.
  4. Output cache folder: feature_cache_node_thresh<PP>/<measure>.

Thresholding: each participant keeps the strongest pct * N(N-1)/2 edges of their own matrix
with weights preserved. If the target exceeds the number of non-zero edges, no edges are
added (the surplus is counted as phantom edges and the density stays below nominal).

Usage:
  python compute_node_features.py 0.20
  python compute_node_features.py 0.20 --measure sift_invnodevol_radius2_count --resolutions 156,1056
  for t in 0.05 0.10 0.15 0.20; do python compute_node_features.py $t; done"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"   # one core per job; parallelism comes from the Pool

import argparse
import json
import multiprocessing as mp
import re
import warnings
from pathlib import Path

import numpy as np

import ml_common as cm

warnings.filterwarnings("ignore")

DATA_ROOT = Path(os.environ.get("SNET_QSIRECON_DIR", cm.BASE / "qsirecon_outputs"))   # QSIRecon connectivity matrices

ap = argparse.ArgumentParser()
ap.add_argument("threshold", type=float, help="between 0 and 1, e.g. 0.20")
ap.add_argument("--measure", default=cm.DEFAULT_MEASURE)
ap.add_argument("--resolutions", default=None, help="comma-separated, e.g. 156,1056")
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
ARGS = ap.parse_args()
PCT = ARGS.threshold
if not (0 < PCT <= 1):
    raise SystemExit(f"Threshold must be between 0 and 1, got: {PCT}")
MEASURE = ARGS.measure
RESOLUTIONS = ([int(x) for x in ARGS.resolutions.split(",")] if ARGS.resolutions
               else cm.RESOLUTIONS)


def find_matrix(sid, res):
    d = DATA_ROOT / f"sub-{sid}"
    if not d.exists():
        raise FileNotFoundError(f"Participant folder not found: {d}")
    pat = re.compile(rf"^4S{res}Parcels_{re.escape(MEASURE)}_connectivity\.mat$")
    m = [f for f in d.rglob("*.mat") if pat.search(f.name)]
    if len(m) != 1:
        raise FileNotFoundError(f"sub-{sid} 4S{res}: {len(m)} matching .mat files (expected 1)")
    return m[0]


def load_matrix(sid, res):
    """Read the raw .mat only; symmetrize in memory and set negatives and the diagonal to zero."""
    import scipy.io as sio
    mat = sio.loadmat(str(find_matrix(sid, res)))
    A = np.asarray(mat[f"atlas_4S{res}Parcels_{MEASURE}_connectivity"], dtype=np.float64)
    A = 0.5 * (A + A.T)
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    return A


def proportional_threshold(A, pct):
    n = A.shape[0]
    iu0, iu1 = np.triu_indices(n, k=1)
    w = A[iu0, iu1]
    n_possible = len(w)
    n_keep = max(0, min(int(round(pct * n_possible)), n_possible))
    n_nonzero = int(np.sum(w > 0))
    phantom = max(0, n_keep - n_nonzero)
    At = np.zeros_like(A)
    if n_keep > 0:
        keep = np.argsort(-w, kind="stable")[:n_keep]
        ii, jj = iu0[keep], iu1[keep]
        At[ii, jj] = A[ii, jj]
        At[jj, ii] = A[ii, jj]
    report = {"density_raw": n_nonzero / n_possible,
              "density_after": int(np.sum(At[iu0, iu1] > 0)) / n_possible,
              "target_edges": n_keep, "phantom_edges": phantom}
    return At, report


def nodal_metrics(A):
    import bct
    n = len(A)
    strength = bct.strengths_und(A)
    wmax = A.max()
    clustering = bct.clustering_coef_wu(A / wmax) if wmax > 0 else np.zeros(n)   # fix (normalized weights)
    eigenvector = bct.eigenvector_centrality_und(A)

    L = bct.weight_conversion(A, "lengths")
    D, _ = bct.distance_wei(L)
    betweenness = bct.betweenness_wei(L)

    with np.errstate(divide="ignore"):
        inv = 1.0 / D
    np.fill_diagonal(inv, 0.0)
    efficiency = inv.sum(axis=1) / (n - 1)

    finite = np.isfinite(D) & ~np.eye(n, dtype=bool)
    reach = finite.sum(axis=1).astype(np.float64)
    sum_d = np.where(finite, D, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        closeness = (reach / (n - 1)) * np.where(sum_d > 0, reach / sum_d, 0.0)
    closeness = np.where(reach > 0, closeness, 0.0)
    return np.column_stack([strength, clustering, betweenness, efficiency, closeness, eigenvector])


def worker(args):
    sid, res = args
    At, rep = proportional_threshold(load_matrix(sid, res), PCT)
    return nodal_metrics(At), rep


def main():
    meta = json.load(open(cm.GLOBAL_CACHE / MEASURE / "meta.json"))
    subjects = meta["subjects"]
    out_dir = cm.node_cache_dir(PCT, MEASURE)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"measure": MEASURE, "subjects": subjects, "labels3": meta["labels3"],
                   "label_names": meta["label_names"], "metric_names": cm.NODE_METRICS,
                   "threshold_pct": PCT,
                   "clustering_note": "clustering_coef_wu on W/max(W) per subject"}, f, indent=2)
    print(f"Threshold: {PCT*100:.0f}% | Measure: {MEASURE} | participants: {len(subjects)} | output: {out_dir}")

    with mp.get_context("fork").Pool(ARGS.workers) as pool:
        for r in RESOLUTIONS:
            out = out_dir / f"X_{r}.npy"
            if out.exists() and (out_dir / f"edge_report_{r}.json").exists():
                print(f"4S{r}: already exists, skipping")
                continue
            print(f"4S{r}: computing...", flush=True)
            results = pool.map(worker, [(s, r) for s in subjects])
            X = np.stack([a for a, _ in results])
            reps = [dict(subject=s, **rep) for s, (_, rep) in zip(subjects, results)]
            if not np.all(np.isfinite(X)):
                print(f"   WARNING: {np.sum(~np.isfinite(X))} non-finite values")
            n_aff = sum(1 for d in reps if d["phantom_edges"] > 0)
            print(f"   shape {X.shape} | participants not reaching target density: {n_aff}/{len(subjects)}"
                  f" | raw density min/median: {min(d['density_raw'] for d in reps):.3f}/"
                  f"{np.median([d['density_raw'] for d in reps]):.3f}")
            np.save(out, X)
            json.dump(reps, open(out_dir / f"edge_report_{r}.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
