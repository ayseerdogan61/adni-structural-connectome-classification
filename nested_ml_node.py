#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nested_ml_node.py - repeated nested CV for node-level classification (observed results).

- Outer CV is repeated N times (default 10). Reported BA/AUC/F1/MCC are means over repeats,
  with the SD across repeats and across all folds.
- Repeat 0 (seed 42) is also stored as 'ba_rep0' so the permutation test can be compared
  with exactly the same pipeline.
- With --covariates, covariates (age, sex, ...) are regressed out within each fold.
- Selection frequencies of the 50 most selected node-metric features are stored.
- Resolution x classifier configurations run in parallel.

Usage:
  python nested_ml_node.py CN_AD 0.20
  python nested_ml_node.py CN_AD 0.20 --covariates covariates.csv --repeats 10 --workers 16
Output: <SNET_BASE>/final_ml/results_node_thresh<PP>/<PAIR>__<measure>[_cov].json"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import ml_common as cm

_DATA = {}


def _job(args):
    r, model_name = args
    X, y, C, names = _DATA["xs"][r], _DATA["y"], _DATA["C"], _DATA["names"][r]
    k_grid = cm.k_grid_for(X.shape[1])
    t0 = time.time()
    o = cm.run_repeated(X, y, C, model_name, k_grid, names, _DATA["repeats"])
    o["seconds"] = round(time.time() - t0, 1)
    return r, model_name, int(X.shape[1]), k_grid, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(cm.PAIRS))
    ap.add_argument("threshold", type=float)
    ap.add_argument("--measure", default=cm.DEFAULT_MEASURE)
    ap.add_argument("--covariates", default=None, help="CSV: subject,age,sex,...")
    ap.add_argument("--repeats", type=int, default=cm.N_REPEATS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args()

    y, xs, names, subjects, C, cov_names = cm.load_pair(a.pair, a.threshold, a.measure, a.covariates)
    _DATA.update(xs=xs, y=y, C=C, names=names, repeats=a.repeats)
    print(f"{a.pair}: n={len(y)} (class0={np.sum(y == 0)}, class1={np.sum(y == 1)}) | "
          f"threshold %{a.threshold*100:.0f} | repeats {a.repeats} | covariates: {cov_names or 'none'}")

    jobs = [(r, m) for r in cm.RESOLUTIONS for m in cm.MODEL_NAMES]
    results = {str(r): {} for r in cm.RESOLUTIONS}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for r, m, n_feat, k_grid, o in pool.imap_unordered(_job, jobs):
            results[str(r)]["n_features"] = n_feat
            results[str(r)]["k_grid"] = k_grid
            results[str(r)][m] = o
            print(f"4S{r:<6}{m:<14}{n_feat:>6}  BA {o['ba']:.3f} (SD_repeats {o['ba_sd_repeats']:.3f})"
                  f"  AUC {o['auc']:.3f}  MCC {o['mcc']:.3f}  k_med {o['k_median']:.0f}"
                  f"  [{o['seconds']} s]", flush=True)

    out = cm.result_path(a.pair, a.threshold, a.measure, a.covariates)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pair": a.pair, "measure": a.measure, "n": int(len(y)),
               "n_class0": int(np.sum(y == 0)), "n_class1": int(np.sum(y == 1)),
               "threshold_pct": a.threshold, "repeats": a.repeats, "seed": cm.SEED,
               "covariates": cov_names, "k_percentages": cm.K_PERCENTAGES, "k_cap": cm.K_CAP,
               "metric_names": cm.NODE_METRICS, "results": results},
              open(out, "w"), indent=1)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
