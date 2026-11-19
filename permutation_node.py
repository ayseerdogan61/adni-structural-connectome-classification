#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""permutation_node.py - label-permutation null distribution over the whole search space.

The best configuration is the maximum over 4 thresholds x 10 resolutions x 3 classifiers
(120 configurations). Because this choice is made outside the nested CV it is optimistic.
Labels are therefore shuffled and the same 120-configuration search is repeated; the
maximum over the 120 configurations is stored for every permutation and compared with the
observed maximum (family-wise p). The same data also give a p-value per configuration and
for the 30-configuration mean at each threshold (see build_excel_report.py).

In each permutation the same shuffled labels are used for all 120 configurations (required
for the family-wise test). Pipeline: ml_common.nested_cv_once with seed 42 (= observed
'ba_rep0'). Covariates stay attached to the participants; only the labels are shuffled.

Runtime: one permutation = 120 configurations x 155 model fits, about 1 min on 32 cores.
At least 1000 permutations are recommended (lower limit 200), split with a SLURM array:
  #SBATCH --array=0-19     -> 50 permutations per job
  python permutation_node.py CN_AD --start $((SLURM_ARRAY_TASK_ID*50)) --n 50

Usage:
  python permutation_node.py CN_AD --start 0 --n 50 [--covariates covariates.csv]
Output: <SNET_BASE>/final_ml/permutations_node/<PAIR>__<measure>[_cov]/perm_<start>_<n>.npz
        with ba[perm, threshold, resolution, model] and auc[...]"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import ml_common as cm

PERM_SEED = 2026
_DATA = {}


def _job(args):
    t_i, r, m_i = args
    X = _DATA["xs"][t_i][r]
    o = cm.nested_cv_once(X, _DATA["yp"], _DATA["C"], cm.MODEL_NAMES[m_i],
                          cm.k_grid_for(X.shape[1]), cm.SEED)
    return t_i, r, m_i, float(np.mean(o["ba"])), float(np.mean(o["auc"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(cm.PAIRS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--measure", default=cm.DEFAULT_MEASURE)
    ap.add_argument("--covariates", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args()

    xs_all, y_ref, C = [], None, None
    for t in cm.THRESHOLDS:
        y, xs, _, subjects, C, _ = cm.load_pair(a.pair, t, a.measure, a.covariates)
        if y_ref is not None and not np.array_equal(y, y_ref):
            raise SystemExit("Participant/label order differs between thresholds")
        y_ref = y
        xs_all.append(xs)
    _DATA.update(xs=xs_all, C=C)

    shape = (a.n, len(cm.THRESHOLDS), len(cm.RESOLUTIONS), len(cm.MODEL_NAMES))
    ba, auc = np.full(shape, np.nan), np.full(shape, np.nan)
    r_idx = {r: i for i, r in enumerate(cm.RESOLUTIONS)}
    jobs = [(t_i, r, m_i) for t_i in range(len(cm.THRESHOLDS))
            for r in cm.RESOLUTIONS for m_i in range(len(cm.MODEL_NAMES))]

    out_dir = cm.perm_dir(a.pair, a.measure, a.covariates)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"perm_{a.start:05d}_{a.n}.npz"

    for p in range(a.n):
        perm_id = a.start + p
        rng = np.random.default_rng(PERM_SEED + perm_id)   # each permutation is reproducible
        _DATA["yp"] = rng.permutation(y_ref)
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:   # new pool per permutation so workers see the current yp
            for t_i, r, m_i, b, u in pool.imap_unordered(_job, jobs):
                ba[p, t_i, r_idx[r], m_i] = b
                auc[p, t_i, r_idx[r], m_i] = u
        print(f"perm {perm_id}: max BA {np.nanmax(ba[p]):.3f} [{time.time() - t0:.0f} s]", flush=True)
        np.savez(out, ba=ba[:p + 1], auc=auc[:p + 1], start=a.start,
                 thresholds=cm.THRESHOLDS, resolutions=cm.RESOLUTIONS, models=cm.MODEL_NAMES)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
