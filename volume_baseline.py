"""volume_baseline.py - volume-only baseline (4S156) with the same nested CV as the main analysis.

Regional volumes replace the network measures as features (absolute volumes and volumes
relative to the total of the 156 regions), for all three tasks. Observed values of the main
analysis (4 thresholds) are printed for comparison.
Run with SNET_BASE set to the analysis folder. Output: ~/volume_baseline_results.json"""

import json, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_common as cm

VOLUME_CSV = os.path.expanduser("~/volumes_4S156.csv")
OUT = os.path.expanduser("~/volume_baseline_results.json")
MAIN = str(cm.FINAL_ML)
THRESHOLD = 0.20   # meta.json is read only for the subject list and labels

def norm(s):
    return str(s).replace("sub-", "")

vol = pd.read_csv(VOLUME_CSV, dtype={"subject": str})
vol["subject"] = vol["subject"].map(norm)
vol = vol.set_index("subject")
reg = [c for c in vol.columns if c.startswith("r")]
print("volume table:", vol.shape, "| region columns:", len(reg))

meta = json.load(open(cm.node_cache_dir(THRESHOLD, cm.DEFAULT_MEASURE) / "meta.json"))
labels3 = np.array(meta["labels3"])
subj_all = [norm(s) for s in meta["subjects"]]
missing = [s for s in subj_all if s not in vol.index]
if missing:
    raise SystemExit(f"Participants missing from volume table: {len(missing)} e.g. {missing[:5]}")

V = vol.loc[subj_all, reg].to_numpy(dtype=float)
VARIANTS = {"raw_volume": V, "relative_volume": V / V.sum(axis=1, keepdims=True)}
names = [f"vol_{c}" for c in reg]

D = {}
for pair, (c0, c1) in cm.PAIRS.items():
    mask = np.isin(labels3, [c0, c1])
    y = (labels3[mask] == c1).astype(int)
    D[pair] = (mask, y)
    print(pair, "n =", len(y), "| class0 =", int((y == 0).sum()), "| class1 =", int((y == 1).sum()))

def job(args):
    pair, var, model = args
    mask, y = D[pair]
    X = VARIANTS[var][mask]
    kg = cm.k_grid_for(X.shape[1])
    o = cm.run_repeated(X, y, None, model, kg, names, cm.N_REPEATS)
    return pair, var, model, kg, o

jobs = [(p, v, m) for p in cm.PAIRS for v in VARIANTS for m in cm.MODEL_NAMES]
results = {}
with mp.get_context("fork").Pool(4) as pool:
    for pair, var, model, kg, o in pool.imap_unordered(job, jobs):
        results.setdefault(pair, {}).setdefault(var, {})[model] = o
        results[pair][var]["k_grid"] = kg
        print(f"done: {pair} {var} {model} BA {o['ba']:.3f}", flush=True)

json.dump(results, open(OUT, "w"), indent=1)

def main_ba(pair, model):
    values = []
    for t in ("05", "10", "15", "20"):
        f = f"{MAIN}/results_node_thresh{t}/{pair}__{cm.DEFAULT_MEASURE}.json"
        try:
            values.append(json.load(open(f))["results"]["156"][model]["ba"])
        except Exception:
            pass
    return (min(values), max(values)) if values else (float("nan"), float("nan"))

print("\n=== VOLUME-BASED (4S156, no covariates) vs MAIN RESULT (4S156 node features, range over 4 thresholds) ===")
print(f"{'task':8}{'model':14}{'absolute BA':>16}{'relative BA':>16}{'abs. AUC':>10}{'main BA min-max':>18}")
for pair in cm.PAIRS:
    for model in cm.MODEL_NAMES:
        h = results[pair]["raw_volume"][model]
        g = results[pair]["relative_volume"][model]
        lo, hi = main_ba(pair, model)
        print(f"{pair:8}{model:14}{h['ba']:>9.3f}±{h['ba_sd_repeats']:.3f}{g['ba']:>9.3f}±{g['ba_sd_repeats']:.3f}"
              f"{h['auc']:>10.3f}{lo:>10.3f}-{hi:.3f}")
print("\nsaved:", OUT)
