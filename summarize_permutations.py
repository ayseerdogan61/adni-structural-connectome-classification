#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_permutations.py - p-values and null-maximum figure from the permutation results.

Input:
  - Observed: final_ml/results_node_thresh<PP>/<PAIR>__<measure>[_cov].json
    ('ba_rep0' = BA of the seed-42 partition)
  - Null: final_ml/permutations_node/<PAIR>__<measure>[_cov]/perm_*.npz

'ba_rep0' is used as the observed value because the null distribution was generated with
the same single partition (seed 42), so both come from exactly the same pipeline. The
10-repeat mean is the main performance estimate; significance comes from these p-values.

Tests (per task):
  1. Family-wise p: maximum BA over the 120 configurations vs the null maximum.
  2. Overall p: mean BA over the 120 configurations vs the null mean.
  3. Threshold-wise p: mean of the 30 configurations at each threshold.
  4. Resolution-wise p: mean of the 12 configurations (4 thresholds x 3 classifiers).
  5. Resolution trend: slope of mean BA on log(number of nodes), two-sided test against the
     null slopes (configurations are not independent, so no Spearman p).
  6. Configuration-wise p: one per configuration (120; supplementary, uncorrected).

Output: final_ml/permutation_summary[_cov]/ with summary_<PAIR>.csv,
        config_p_<PAIR>.csv, summary_all.txt and fig_null_max.png/.pdf

Usage: python summarize_permutations.py [--measure ...] [--use-cov-results] [--min-perm 100]"""

import argparse
import json

import numpy as np

import ml_common as cm

ap = argparse.ArgumentParser()
ap.add_argument("--measure", default=cm.DEFAULT_MEASURE)
ap.add_argument("--use-cov-results", action="store_true")
ap.add_argument("--min-perm", type=int, default=100, help="warn if fewer permutations than this")
a = ap.parse_args()
COV = a.use_cov_results
T, R, M = cm.THRESHOLDS, cm.RESOLUTIONS, cm.MODEL_NAMES
LOGR = np.log(np.array(R, dtype=float))


def p_upper(null, obs):
    """(1 + #null >= obs) / (1 + B)  [Phipson & Smyth 2010]"""
    return (1 + np.sum(null >= obs - 1e-12)) / (1 + len(null))


def p_two_sided(null, obs):
    c = np.mean(null)
    return (1 + np.sum(np.abs(null - c) >= abs(obs - c) - 1e-12)) / (1 + len(null))


def slope(ba_tr):
    """ba_tr: (..., T, R, M) -> slope of the per-resolution mean BA on log(N)."""
    per_res = ba_tr.mean(axis=(-3, -1))                     # (..., R)
    x = LOGR - LOGR.mean()
    return (per_res - per_res.mean(axis=-1, keepdims=True)) @ x / (x @ x)


def load_observed(pair):
    obs0 = np.full((len(T), len(R), len(M)), np.nan)
    obs = np.full_like(obs0, np.nan)
    for i, t in enumerate(T):
        d = json.load(open(cm.result_path(pair, t, a.measure, COV)))["results"]
        for j, r in enumerate(R):
            for k, m in enumerate(M):
                obs0[i, j, k] = d[str(r)][m]["ba_rep0"]
                obs[i, j, k] = d[str(r)][m]["ba"]
    return obs0, obs


def load_null(pair):
    d = cm.perm_dir(pair, a.measure, COV)
    files = sorted(d.glob("perm_*.npz")) if d.exists() else []
    if not files:
        return None
    parts, starts = [], []
    for f in files:
        z = np.load(f)
        if list(z["resolutions"]) != R or len(z["thresholds"]) != len(T):
            raise SystemExit(f"{f}: threshold/resolution list does not match the observed results")
        parts.append(z["ba"])
        starts.append(int(z["start"]))
    if len(set(starts)) != len(starts):
        raise SystemExit(f"{pair}: several files share the same --start (double counting)")
    ba = np.concatenate(parts)
    return ba[~np.isnan(ba).any(axis=(1, 2, 3))]


out_dir = cm.FINAL_ML / f"permutation_summary{'_cov' if COV else ''}"
out_dir.mkdir(parents=True, exist_ok=True)
lines, fig_data = [], {}

for pair in cm.PAIRS:
    try:
        obs0, obs = load_observed(pair)
    except FileNotFoundError as e:
        print(f"{pair}: observed result missing ({e.filename}), skipping")
        continue
    null = load_null(pair)
    if null is None:
        print(f"{pair}: no permutation files, skipping")
        continue
    B = len(null)
    warn = f"  WARNING: only {B} permutations" if B < a.min_perm else ""
    fig_data[pair] = (null.max(axis=(1, 2, 3)), obs0.max())

    rows = []   # (test, observed repeat 0, observed 10-repeat, null mean, null 95th percentile, p)

    def add(name, o0, o, nl, p):
        rows.append((name, o0, o, float(np.mean(nl)), float(np.percentile(nl, 95)), p))

    nmax = null.max(axis=(1, 2, 3))
    i, j, k = np.unravel_index(np.argmax(obs0), obs0.shape)
    add(f"MAX of 120 configs (best at repeat 0: {T[i]*100:.0f} 4S{R[j]} {M[k]})",
        obs0.max(), obs[i, j, k], nmax, p_upper(nmax, obs0.max()))
    nmean = null.mean(axis=(1, 2, 3))
    add("MEAN of 120 configs", obs0.mean(), obs.mean(), nmean, p_upper(nmean, obs0.mean()))
    for ti, t in enumerate(T):
        nl = null[:, ti].mean(axis=(1, 2))
        add(f"Threshold {t*100:.0f}% (mean of 30 configs)", obs0[ti].mean(), obs[ti].mean(), nl,
            p_upper(nl, obs0[ti].mean()))
    for ri, r in enumerate(R):
        nl = null[:, :, ri].mean(axis=(1, 2))
        add(f"Resolution 4S{r} (mean of 12 configs)", obs0[:, ri].mean(), obs[:, ri].mean(), nl,
            p_upper(nl, obs0[:, ri].mean()))
    ns = slope(null)
    add("Resolution trend (BA / log N), two-sided", slope(obs0), slope(obs), ns,
        p_two_sided(ns, slope(obs0)))

    hdr = "test,observed_repeat0,observed_10repeat,null_mean,null_95th_percentile,p"
    with open(out_dir / f"summary_{pair}.csv", "w") as f:
        f.write(hdr + "\n")
        for r_ in rows:
            f.write(f"\"{r_[0]}\",{r_[1]:.4f},{r_[2]:.4f},{r_[3]:.4f},{r_[4]:.4f},{r_[5]:.4f}\n")
    with open(out_dir / f"config_p_{pair}.csv", "w") as f:
        f.write("threshold,resolution,model,ba_repeat0,ba_10repeat,p_uncorrected\n")
        for ti, t in enumerate(T):
            for ri, r in enumerate(R):
                for mi, m in enumerate(M):
                    p = p_upper(null[:, ti, ri, mi], obs0[ti, ri, mi])
                    f.write(f"{t},{r},{m},{obs0[ti, ri, mi]:.4f},{obs[ti, ri, mi]:.4f},{p:.4f}\n")

    lines.append(f"\n=== {pair}  |  {B} permutations{warn}")
    lines.append(f"{'test':<58}{'obs(t0)':>9}{'obs(10)':>9}{'null mean':>9}{'null 95%':>9}{'p':>8}")
    for r_ in rows:
        lines.append(f"{r_[0]:<58}{r_[1]:>9.3f}{r_[2]:>9.3f}{r_[3]:>9.3f}{r_[4]:>9.3f}{r_[5]:>8.4f}")
    lines.append(f"  (smallest attainable p = {1/(B+1):.4f})")

text = "\n".join(lines)
print(text)
(out_dir / "summary_all.txt").write_text(text + "\n")

# ---------------- Figure: null distributions of the maximum
if fig_data:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"CN_AD": "CN vs AD", "CN_MCI": "CN vs MCI", "MCI_AD": "MCI vs AD"}
    fig, axes = plt.subplots(1, len(fig_data), figsize=(4.2 * len(fig_data), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (pair, (nmax, o)), lab in zip(axes, fig_data.items(), "ABC"):
        ax.hist(nmax, bins=30, color="#9aa7b8", edgecolor="white")
        ax.axvline(o, color="#c0392b", lw=2)
        ax.axvline(np.percentile(nmax, 95), color="#555", lw=1, ls="--")
        p = p_upper(nmax, o)
        ax.set_title(f"({lab}) {titles[pair]}", loc="left", fontsize=11)
        ax.text(0.98, 0.95, f"observed max = {o:.3f}\np = {p:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        ax.set_xlabel("Maximum balanced accuracy\n(120 configurations)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Permutations")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_null_max.{ext}", dpi=300)
    print(f"\nFigure: {out_dir / 'fig_null_max.png'}")
print(f"Outputs: {out_dir}")
