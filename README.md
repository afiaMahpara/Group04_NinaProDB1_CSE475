# NinaPro DB1 Gesture Classification 

**Group 04 — CSE475**

Classifying hand and finger gestures from surface EMG (and glove kinematic) signals
using the [NinaPro DB1](https://ninapro.hevs.ch/instructions/DB1.html) dataset,
comparing classical machine learning baselines against a proposed Graph Neural
Network (GNN), with a systematic architecture ablation study.

## Problem Statement

NinaPro DB1 records 27 intact subjects performing 52 hand/finger gestures (plus
rest) across 3 exercises, using 10 EMG electrodes and a 22-sensor Cyberglove at
100 Hz. We ask: **does representing the sensor array as a graph — and applying a
GNN — outperform classical ML models that treat sensor channels as a flat feature
vector?**

## Results Summary

| Task | Best Model | Macro-F1 | Notes |
|---|---|---|---|
| Task 2(a) — Baselines | SVM (RBF kernel) | **0.7842** | 23 classes, EMG+glove, all 3 exercises |
| Task 2(b) — Proposed GNN | EdgeRefineGCN | 0.3945 | Same scope as baselines |
| Task 3 — Ablation, final GNN | GAT (5-fold CV) | 0.1433 ± 0.0142 | Reduced scope: Exercise A only, EMG-only, 13 classes |
| Task 3 — Paired baseline (same folds) | Random Forest | 0.2238 ± 0.0369 | For fair comparison against the final GNN |

**Headline finding:** the proposed GNN does not outperform classical baselines in
either task. We diagnose this honestly rather than hide it — see
[`report/`](report/) for the full analysis, including a cross-exercise **label
collision** we found in our own EDA (Task 1) that measurably hurts the GNN more
than the classical models.

## Repository Structure

```
.
├── code/
│   ├── task1/    → EDA notebook (dataset characterization)
│   ├── task2/    → Baseline models + proposed GNN notebooks
│   └── task3/    → Ablation study notebook
├── models/       → Trained model artifacts (see models/README.md)
├── papers/       → Reference papers used for background/methodology
├── related_work/ → The 5 related-work papers used for Pillar A comparison
├── report/
│   ├── task1/    → EDA report
│   ├── task2/    → Baseline models + proposed GNN report
│   └── task3/    → Ablation study report
└── README.md    
```

## Task-by-Task Overview

### Task 1 — Exploratory Data Analysis
`code/task1/Group04_NinaProDB1_task1_eda.ipynb`

Characterizes the dataset before modeling: summary statistics, data-quality checks
(missing values, duplicates, redundant features), class balance, feature
distributions, correlation structure, and PCA/t-SNE/UMAP projections. The key
finding — a **cross-exercise gesture label collision** (DB1 numbers gestures
locally within each exercise, so combining all 3 exercises under one label column
causes classes 1–17 to mix two different physical gestures each) — is the most
consequential structural property of the dataset for everything downstream.

### Task 2 — Baselines and Proposed GNN
`code/task2/`

- **Baselines:** 8 classical models (SVM, Random Forest, LightGBM, XGBoost, MLP,
  k-NN, Logistic Regression, Decision Tree) trained on a subject-independent split
  (subjects 1–20 train / 21–27 test), with SMOTE for class balancing and
  feature selection to the top 100 (of 128) features. **SVM (RBF) wins**,
  Macro-F1 = 0.7842.
- **Proposed GNN (EdgeRefineGCN):** represents all 32 sensor channels as graph
  nodes (4 features each: RMS, MAV, WL, VAR), with edges from training-only
  channel correlation, refined by a small learnable MLP. 3-layer GCN, 10,544
  parameters. Achieves Macro-F1 = 0.3945 — underperforming every baseline.

### Task 3 — Ablation Study
`code/task3/`

Due to compute-time constraints, this task uses a **reduced scope** relative to
Task 2: Exercise A only, EMG-only (10 nodes, not 32), 13 classes including rest.
Systematically tests 8 architectural/training axes (layers, hidden dimension,
dropout, graph construction, architecture variant, edge weighting, pooling,
training strategy) across 24 configurations, selects a final configuration, and
validates it with 5-fold subject-wise cross-validation plus a paired Wilcoxon
signed-rank significance test against a same-folds Random Forest baseline. The
final GNN (Macro-F1 = 0.1433 ± 0.0142) does not beat the paired baseline
(0.2238 ± 0.0369) on any of the 5 folds, though the difference does not reach
formal significance at this fold count (p = 0.0625).

## Data Leakage Controls

Every experiment in this project uses **subject-wise** splitting — never random
row-wise splitting — so that a subject's data never appears in both training and
test/validation sets. This is enforced with explicit assertions in every notebook
(e.g. `assert train_subjects.isdisjoint(test_subjects)`), including inside the
5-fold `GroupKFold` cross-validation in Task 3.

## Setup / Reproducing

All notebooks are designed to run on Kaggle with the DB1 dataset added as an input
(update `DATA_PATH` / `DATASET_PATH` in each notebook's configuration cell to match
your dataset's mount path). Requirements: PyTorch, PyTorch Geometric, scikit-learn,
XGBoost, LightGBM (optional), pandas, numpy, matplotlib/seaborn, scipy.

```bash
pip install torch torch_geometric scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn scipy
```

## Reports

Full written reports for each task include complete
methodology, results tables, figures, and — critically — an honest discussion of
why the proposed GNN underperforms classical baselines in this project, connecting
the result back to the Task 1 EDA's label-collision finding.


