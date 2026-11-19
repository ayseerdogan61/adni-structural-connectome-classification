"""volume_adjusted.py - volume-adjusted classification (CN vs AD, 4S156).

The volume of each node is regressed out of that node's own network measures within the
training folds (coefficients from training data only) before feature selection and
classification; everything else is identical to the main analysis.
Run with SNET_BASE set to the analysis folder. Output: ~/volume_adjusted_results.json"""

import glob, json, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd
import scipy.io as sio

os.environ["SNET_RESOLUTIONS"] = "156"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_common as cm

PAIR = "CN_AD"
VOLUME_CSV = os.path.expanduser("~/volumes_4S156.csv")
OUT = os.path.expanduser("~/volume_adjusted_results.json")
MAIN = str(cm.FINAL_ML)

def norm(s):
    return str(s).replace("sub-", "")

# check region order: matrix index must follow region IDs 1..156
example = sorted(glob.glob(os.environ.get("SNET_QSIRECON_DIR", f"{cm.BASE}/qsirecon_outputs") + "/sub-*/4S156Parcels_radius2_count_connectivity.mat"))[0]
ids = sio.loadmat(example)["atlas_4S156Parcels_region_ids"].ravel().astype(int)
assert (ids == np.arange(1, 157)).all(), "region ID order is not 1..156"

vol = pd.read_csv(VOLUME_CSV, dtype={"subject": str})
vol["subject"] = vol["subject"].map(norm)
vol = vol.set_index("subject")
reg = [f"r{i}" for i in ids]

def residualize_node(X_tr, X_te, C_tr, C_te):
    """Remove only the node's own volume from each of its features; coefficients from the training data only."""
    if C_tr is None:
        return X_tr, X_te
    mv, mx = C_tr.mean(0), X_tr.mean(0)
    vc, xc = C_tr - mv, X_tr - mx
    var = (vc ** 2).sum(0)
    b = np.where(var > 0, (vc * xc).sum(0) / np.where(var > 0, var, 1.0), 0.0)
    return xc - vc * b, (X_te - mx) - (C_te - mv) * b

cm.residualize = residualize_node   # all fold operations inside cm use this

D = {}
for pct in cm.THRESHOLDS:
    y, xs, names, subjects, C, _ = cm.load_pair(PAIR, pct, cm.DEFAULT_MEASURE, None)
    X = xs[156]
    missing = [s for s in subjects if norm(s) not in vol.index]
    if missing:
        raise SystemExit(f"participants missing from volume table: {missing[:5]}")
    V = vol.loc[[norm(s) for s in subjects], reg].to_numpy(dtype=float)
    Vf = np.tile(V, (1, len(cm.NODE_METRICS)))   # feature order: metric*156 + node
    assert Vf.shape == X.shape, (Vf.shape, X.shape)
    D[pct] = (y, X, names[156], Vf)
    print(f"threshold {int(pct*100)}%: n={len(y)} | features={X.shape[1]}")

def job(args):
    pct, model = args
    y, X, names, Vf = D[pct]
    kg = cm.k_grid_for(X.shape[1])
    o = cm.run_repeated(X, y, Vf, model, kg, names, cm.N_REPEATS)
    return pct, model, o

jobs = [(p, m) for p in cm.THRESHOLDS for m in cm.MODEL_NAMES]
results = {}
with mp.get_context("fork").Pool(4) as pool:
    for pct, model, o in pool.imap_unordered(job, jobs):
        results.setdefault(str(pct), {})[model] = o
        print(f"done: threshold {pct} {model} BA {o['ba']:.3f}", flush=True)

json.dump(results, open(OUT, "w"), indent=1)

print("\n=== CN-AD, 4S156: unadjusted (main result) vs volume-adjusted ===")
print(f"{'thr.':6}{'model':14}{'unadjusted BA':>16}{'volume-adj. BA':>22}{'diff':>8}{'AUC(adj.)':>11}")
for pct in cm.THRESHOLDS:
    f = f"{MAIN}/results_node_{cm.thresh_tag(pct)}/{PAIR}__{cm.DEFAULT_MEASURE}.json"
    main = json.load(open(f))["results"]["156"]
    for model in cm.MODEL_NAMES:
        a = main[model]["ba"]
        d = results[str(pct)][model]
        print(f"%{int(pct*100):<5}{model:14}{a:>16.3f}{d['ba']:>14.3f}±{d['ba_sd_repeats']:.3f}{d['ba']-a:>+8.3f}{d['auc']:>11.3f}")
print("\nsaved:", OUT)
