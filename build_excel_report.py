#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_excel_report.py - collects the results in one Excel file (no model training).

Sheets:
  Summary              per task x threshold: mean BA of the 30 configurations (+ permutation p),
                       best configuration (repeated BA +- SD), family-wise permutation p of the
                       maximum over the 120 configurations
  CN_AD / CN_MCI / MCI_AD   per threshold, one row per resolution x classifier: median k,
                       mean BA +- SD, AUC, F1, MCC, BA (repeat 0), per-configuration p
  Selected_Features    20 most frequently selected node-metric features of the best configuration
  Density              per threshold x resolution: participants not reaching the target density,
                       raw density min / median
  Demographics         (with --covariates) group means and pairwise group tests

If permutation files are missing the p columns stay empty; the report is still produced.

Usage: python build_excel_report.py [--measure ...] [--covariates covariates.csv]
                                       [--use-cov-results]
Output: <SNET_BASE>/final_ml/Classification_Results[_cov].xlsx"""

import argparse
import json

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

import ml_common as cm

ap = argparse.ArgumentParser()
ap.add_argument("--measure", default=cm.DEFAULT_MEASURE)
ap.add_argument("--covariates", default=None, help="Demografi sayfasi icin CSV")
ap.add_argument("--use-cov-results", action="store_true")
a = ap.parse_args()
COV = a.use_cov_results
GROUPS = ["CN_AD", "CN_MCI", "MCI_AD"]
T, R, M = cm.THRESHOLDS, cm.RESOLUTIONS, cm.MODEL_NAMES
BOLD, HFONT = Font(bold=True), Font(bold=True, size=12)


def load_results(pair):
    out = {}
    for t in T:
        p = cm.result_path(pair, t, a.measure, COV)
        if not p.exists():
            print(f"Warning: {p} not found")
            return None
        out[t] = json.load(open(p))
    return out


def load_perms(pair):
    d = cm.perm_dir(pair, a.measure, COV)
    files = sorted(d.glob("perm_*.npz")) if d.exists() else []
    if not files:
        return None
    ba = np.concatenate([np.load(f)["ba"] for f in files])
    ba = ba[~np.isnan(ba).any(axis=(1, 2, 3))]
    return ba if len(ba) else None


def p_ge(null, obs):
    return (1 + np.sum(null >= obs - 1e-12)) / (1 + len(null))


def fmt_p(p):
    return None if p is None else round(float(p), 4)


wb = Workbook()
ws_sum = wb.active
ws_sum.title = "Summary"
sum_head = ["Task", "n (0/1)", "Threshold", "Mean BA (30 configs, repeated)", "Mean BA repeat 0",
            "p (mean)", "Best config", "Best BA +- SD", "Max BA repeat 0 (120 configs)",
            "p family-wise (max)", "Number of permutations"]
for j, h in enumerate(sum_head, 1):
    ws_sum.cell(row=1, column=j, value=h).font = BOLD
srow = 2
ws_feat = wb.create_sheet("Selected_Features")
frow = 1

for pair in GROUPS:
    res = load_results(pair)
    if res is None:
        continue
    null = load_perms(pair)                         # (P, T, R, M) or None
    obs0 = np.array([[[res[t]["results"][str(r)][m]["ba_rep0"] for m in M] for r in R] for t in T])
    obs = np.array([[[res[t]["results"][str(r)][m]["ba"] for m in M] for r in R] for t in T])
    p_fw = p_ge(null.max(axis=(1, 2, 3)), obs0.max()) if null is not None else None
    n_perm = 0 if null is None else len(null)
    info = res[T[0]]

    # ---------------- Summary
    for t_i, t in enumerate(T):
        p_mean = p_ge(null[:, t_i].mean(axis=(1, 2)), obs0[t_i].mean()) if null is not None else None
        r_i, m_i = np.unravel_index(np.argmax(obs[t_i]), obs[t_i].shape)
        best = res[t]["results"][str(R[r_i])][M[m_i]]
        row = [pair, f"{info['n']} ({info['n_class0']}/{info['n_class1']})", f"%{t*100:.0f}",
               round(obs[t_i].mean(), 3), round(obs0[t_i].mean(), 3), fmt_p(p_mean),
               f"4S{R[r_i]} {M[m_i]}", f"{best['ba']:.3f} +- {best['ba_sd_repeats']:.3f}",
               round(obs0.max(), 3) if t_i == 0 else None, fmt_p(p_fw) if t_i == 0 else None,
               n_perm if t_i == 0 else None]
        for j, v in enumerate(row, 1):
            ws_sum.cell(row=srow, column=j, value=v)
        srow += 1
    srow += 1

    # ---------------- Task sheet
    ws = wb.create_sheet(pair)
    head = ["Feature set (count)", "Model", "k median", "BA mean", "BA SD (repeats)", "BA SD (folds)",
            "AUC", "F1", "MCC", "BA repeat 0", "p (config)"]
    max_ba = obs.max()
    row = 1
    for t_i, t in enumerate(T):
        ws.cell(row=row, column=1, value=f"THRESHOLD: {t*100:.0f}%  (repeats={res[t]['repeats']}, "
                                         f"covariates={res[t]['covariates'] or 'none'})").font = HFONT
        row += 1
        for j, h in enumerate(head, 1):
            ws.cell(row=row, column=j, value=h).font = BOLD
        row += 1
        for r_i, r in enumerate(R):
            block = row
            for m_i, m in enumerate(M):
                o = res[t]["results"][str(r)][m]
                p_c = p_ge(null[:, t_i, r_i, m_i], o["ba_rep0"]) if null is not None else None
                vals = [f"4S{r} ({res[t]['results'][str(r)]['n_features']})" if m_i == 0 else None,
                        m, o["k_median"], round(o["ba"], 3), round(o["ba_sd_repeats"], 3),
                        round(o["ba_sd_folds"], 3), round(o["auc"], 3), round(o["f1"], 3),
                        round(o["mcc"], 3), round(o["ba_rep0"], 3), fmt_p(p_c)]
                for j, v in enumerate(vals, 1):
                    c = ws.cell(row=row, column=j, value=v)
                    if j == 4 and o["ba"] == max_ba:
                        c.font = BOLD
                row += 1
            ws.merge_cells(start_row=block, start_column=1, end_row=block + len(M) - 1, end_column=1)
            ws.cell(row=block, column=1).alignment = Alignment(vertical="center")
        row += 1
    for j in range(1, len(head) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 20 if j == 1 else 13

    # ---------------- Selected features (best configuration of the task)
    t_i, r_i, m_i = np.unravel_index(np.argmax(obs), obs.shape)
    o = res[T[t_i]]["results"][str(R[r_i])][M[m_i]]
    ws_feat.cell(row=frow, column=1, value=f"{pair}: best config = {T[t_i]*100:.0f}% threshold, "
                 f"4S{R[r_i]}, {M[m_i]} (BA {o['ba']:.3f}); frequency = share of outer folds selected "
                 f"({o['n_outer_folds_total']} fold)").font = HFONT
    frow += 1
    for j, h in enumerate(["Rank", "Feature (metric_node)", "Selection frequency"], 1):
        ws_feat.cell(row=frow, column=j, value=h).font = BOLD
    frow += 1
    for i, d in enumerate(o["selection_freq_top"][:20], 1):
        for j, v in enumerate([i, d["feature"], round(d["freq"], 3)], 1):
            ws_feat.cell(row=frow, column=j, value=v)
        frow += 1
    frow += 1
    print(f"{pair}: done (permutations: {n_perm})")

ws_feat.column_dimensions["B"].width = 30

# ---------------- Density / phantom edges
ws_d = wb.create_sheet("Density")
for j, h in enumerate(["Threshold", "Resolution", "Participants below target density", "Total participants",
                       "Raw density min", "Raw density median", "Mean phantom edges (affected)"], 1):
    ws_d.cell(row=1, column=j, value=h).font = BOLD
drow = 2
for t in T:
    for r in R:
        p = cm.node_cache_dir(t, a.measure) / f"edge_report_{r}.json"
        if not p.exists():
            continue
        rep = json.load(open(p))
        aff = [d["phantom_edges"] for d in rep if d["phantom_edges"] > 0]
        dens = [d["density_raw"] for d in rep]
        vals = [f"%{t*100:.0f}", f"4S{r}", len(aff), len(rep), round(min(dens), 4),
                round(float(np.median(dens)), 4), round(float(np.mean(aff)), 1) if aff else 0]
        for j, v in enumerate(vals, 1):
            ws_d.cell(row=drow, column=j, value=v)
        drow += 1

# ---------------- Demographics
if a.covariates:
    import pandas as pd
    from scipy import stats
    meta = json.load(open(cm.node_cache_dir(T[0], a.measure) / "meta.json"))
    names = {0: "CN", 1: "MCI", 2: "AD"}
    df = pd.read_csv(a.covariates, dtype={"subject": str}).set_index("subject").loc[meta["subjects"]]
    df["group"] = [names[l] for l in meta["labels3"]]
    ws_g = wb.create_sheet("Demographics")
    cols = [c for c in df.columns if c != "group"]
    heads = ["Variable", "CN", "MCI", "AD", "p CN-AD", "p CN-MCI", "p MCI-AD", "Test"]
    for j, h in enumerate(heads, 1):
        ws_g.cell(row=1, column=j, value=h).font = BOLD
    for i, c in enumerate(cols, 2):
        numeric = pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2
        cells = [c]
        for g in ["CN", "MCI", "AD"]:
            v = df.loc[df.group == g, c]
            cells.append(f"{v.mean():.1f} +- {v.std():.1f}" if numeric
                         else ", ".join(f"{k}:{n}" for k, n in v.value_counts().sort_index().items()))
        for g0, g1 in [("CN", "AD"), ("CN", "MCI"), ("MCI", "AD")]:
            sub = df[df.group.isin([g0, g1])]
            if numeric:
                p = stats.ttest_ind(sub.loc[sub.group == g0, c], sub.loc[sub.group == g1, c],
                                    equal_var=False).pvalue
            else:
                p = stats.chi2_contingency(pd.crosstab(sub.group, sub[c]))[1]
            cells.append(round(float(p), 4))
        cells.append("Welch t" if numeric else "Chi-square")
        for j, v in enumerate(cells, 1):
            ws_g.cell(row=i, column=j, value=v)

wb.move_sheet("Selected_Features", offset=len(wb.sheetnames))
for j in range(1, 12):
    ws_sum.column_dimensions[get_column_letter(j)].width = 16
out = cm.FINAL_ML / f"Classification_Results{'_cov' if COV else ''}.xlsx"
out.parent.mkdir(parents=True, exist_ok=True)
wb.save(out)
print(f"Excel hazir: {out}")
