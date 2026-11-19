# Analytic flexibility in structural connectome classification of Alzheimer's disease

Code for node-level, thresholded structural-connectome classification (CN / MCI / AD) on ADNI data
using the 4S atlas family (156–1056 regions), plus the regional-volume control analyses.
**No ADNI data are included**; access must be requested from ADNI (adni.loni.usc.edu).

## Pipeline (run in this order)

1. `compute_node_features.py` – thresholds each participant's connectivity matrix and computes the 6 nodal measures (the features); uses `ml_common.py`, the shared settings and functions imported by all other scripts.
2. `nested_ml_node.py` – trains and evaluates the classifiers with repeated nested cross-validation (the observed results).
3. `permutation_node.py` – repeats the whole 120-configuration search with shuffled labels to build the null distribution.
4. `summarize_permutations.py` – turns the observed and permuted results into p-values and the null-maximum figure.
5. `build_excel_report.py` – collects all results into one Excel report (no model training).
6. `volumes_4S156.py` – extracts regional volumes (mm³) of the 156 regions from the subject-space segmentations.
7. `volume_baseline.py` – classifies with regional volumes alone, to test how much the volume-only signal explains.
8. `volume_adjusted.py` – removes each region's own volume from its network measures (training folds only) and repeats the CN vs AD classification.

## Usage

```bash
# 1) features (one run per threshold)
for t in 0.05 0.10 0.15 0.20; do python compute_node_features.py $t; done

# 2) observed results (3 tasks x 4 thresholds); optional covariates (age, sex)
python nested_ml_node.py CN_AD 0.20 --repeats 10
python nested_ml_node.py CN_AD 0.20 --repeats 10 --covariates covariates.csv

# 3) permutations (>= 200, preferably 1000); e.g. SLURM array of 20 jobs x 50
python permutation_node.py CN_AD --start $((SLURM_ARRAY_TASK_ID*50)) --n 50

# 4) summary and 5) report
python summarize_permutations.py
python build_excel_report.py --covariates covariates.csv
```

Tasks: `CN_AD`, `CN_MCI`, `MCI_AD`. Covariate file: CSV with columns `subject,age,sex`
(`subject` must match the IDs in `meta.json`; text columns such as sex are coded 0/1).

## Design summary

- 10 resolutions (156–1056 regions), 4 proportional thresholds (5/10/15/20 %), 6 nodal measures
  (strength, clustering, betweenness, efficiency, closeness, eigenvector centrality).
- Classifiers: logistic regression, random forest, RBF-SVM. Nested 5×5 stratified CV,
  repeated 10 times (seeds 42–51). Features ranked by Welch t-test on training data only;
  number of features chosen in the inner CV.
- Permutation test over the whole 4 × 10 × 3 = 120-configuration search (family-wise p).
- Covariate and volume adjustment are fitted inside the training folds only.

## Notes

- Folders are set with environment variables: `SNET_BASE` (analysis folder, default `./snet_data`),
  `SNET_QSIRECON_DIR` (connectivity matrices, default `$SNET_BASE/qsirecon_outputs`) and
  `SNET_SEG_DIR` (subject-space segmentations for `volumes_4S156.py`, default `$SNET_BASE/qsirecon_segmentations`).
  `SNET_RESOLUTIONS` (comma-separated) restricts the resolutions, for testing only.
- `compute_node_features.py` expects `feature_cache/<measure>/meta.json` (subject list and labels)
  and the connectivity matrices (`4S<res>Parcels_<measure>_connectivity.mat`) produced by QSIRecon;
  these files are not part of this repository.
- Tested with Python 3.10, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, bctpy 0.6.1,
  openpyxl 3.1.5, matplotlib 3.10.9 and nibabel 5.4.2.
