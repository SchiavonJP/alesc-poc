# Graph Report - ./example/  (2026-04-18)

## Corpus Check
- Corpus is ~1,009 words - fits in a single context window. You may not need a graph.

## Summary
- 29 nodes · 35 edges · 6 communities detected
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_PyCaret Pipeline Core|PyCaret Pipeline Core]]
- [[_COMMUNITY_Dataset & Contamination Estimation|Dataset & Contamination Estimation]]
- [[_COMMUNITY_Notebook Infrastructure|Notebook Infrastructure]]
- [[_COMMUNITY_Candidate Algorithm Pool|Candidate Algorithm Pool]]
- [[_COMMUNITY_Results & Visualization|Results & Visualization]]
- [[_COMMUNITY_Deep Learning Detector|Deep Learning Detector]]

## God Nodes (most connected - your core abstractions)
1. `PyCaret Anomaly Detection Pipeline Notebook` - 13 edges
2. `PyCaret Anomaly Detection Module` - 8 edges
3. `PyOD IsolationForest (IForest)` - 4 edges
4. `Brazilian Congress Expenses Dataset (Ano-2019-2023.csv)` - 3 edges
5. `PyCaret assign_model()` - 3 edges
6. `IQR-based Outlier Detection` - 3 edges
7. `Contamination Fraction Parameter` - 3 edges
8. `scikit-learn IsolationForest` - 2 edges
9. `PyOD Gaussian Mixture Model (GMM)` - 2 edges
10. `PyOD AutoEncoder` - 2 edges

## Surprising Connections (you probably didn't know these)
- `ydata-profiling ProfileReport` --semantically_similar_to--> `IQR-based Outlier Detection`  [INFERRED] [semantically similar]
  example/pipe-pycaret.md → example/pipe-pycaret.md  _Bridges community 2 → community 1_
- `PyCaret Anomaly Detection Pipeline Notebook` --references--> `PyCaret Anomaly Detection Module`  [EXTRACTED]
  example/pipe-pycaret.md → example/pipe-pycaret.md  _Bridges community 2 → community 0_
- `PyCaret Anomaly Detection Pipeline Notebook` --references--> `PyOD IsolationForest (IForest)`  [EXTRACTED]
  example/pipe-pycaret.md → example/pipe-pycaret.md  _Bridges community 2 → community 3_
- `PyCaret Anomaly Detection Pipeline Notebook` --references--> `PyOD AutoEncoder`  [EXTRACTED]
  example/pipe-pycaret.md → example/pipe-pycaret.md  _Bridges community 2 → community 5_
- `PyCaret Anomaly Detection Pipeline Notebook` --references--> `Silhouette Score (sklearn.metrics)`  [EXTRACTED]
  example/pipe-pycaret.md → example/pipe-pycaret.md  _Bridges community 2 → community 4_

## Hyperedges (group relationships)
- **Full Anomaly Detection Pipeline: Setup -> Train -> Assign -> Evaluate** — pipepycaret_anomaly_setup, pipepycaret_create_model, pipepycaret_assign_model, pipepycaret_evaluate_model [EXTRACTED 1.00]
- **Contamination Estimation: IQR Outlier Detection -> Fraction -> IForest Training** — pipepycaret_iqr_outlier_detection, pipepycaret_fraction_contamination, pipepycaret_iforest_pyod [EXTRACTED 0.95]
- **Candidate Anomaly Detection Models (GMM, AutoEncoder, IForest)** — pipepycaret_gmm_pyod, pipepycaret_autoencoder_pyod, pipepycaret_iforest_pyod [EXTRACTED 0.90]

## Communities

### Community 0 - "PyCaret Pipeline Core"
Cohesion: 0.22
Nodes (6): PyCaret Anomaly Setup (setup() call), PCA Preprocessing in PyCaret Setup, PyCaret plot_model() (tsne/umap), PyCaret Anomaly Detection Module, Rationale: PCA enabled in setup for high-cardinality categorical encoding, UMAP Dimensionality Reduction

### Community 1 - "Dataset & Contamination Estimation"
Cohesion: 0.33
Nodes (6): Expense Category: Combustíveis e Lubrificantes, PyCaret create_model(), Brazilian Congress Expenses Dataset (Ano-2019-2023.csv), Contamination Fraction Parameter, IQR-based Outlier Detection, Rationale: Using IQR fraction as contamination parameter

### Community 2 - "Notebook Infrastructure"
Cohesion: 0.4
Nodes (5): joblib (load/dump), Python Logging Configuration, PyCaret Anomaly Detection Pipeline Notebook, ROC AUC Score (sklearn.metrics), ydata-profiling ProfileReport

### Community 3 - "Candidate Algorithm Pool"
Cohesion: 0.5
Nodes (4): PyOD Gaussian Mixture Model (GMM), PyOD IsolationForest (IForest), scikit-learn IsolationForest, PyOD Median Absolute Deviation (MAD)

### Community 4 - "Results & Visualization"
Cohesion: 0.67
Nodes (3): PyCaret assign_model(), Plotly Scatter Plot of Anomaly Scores, Silhouette Score (sklearn.metrics)

### Community 5 - "Deep Learning Detector"
Cohesion: 1.0
Nodes (2): PyOD AutoEncoder, TensorFlow oneDNN/CUDA Warning

## Knowledge Gaps
- **8 isolated node(s):** `ROC AUC Score (sklearn.metrics)`, `Expense Category: Combustíveis e Lubrificantes`, `Plotly Scatter Plot of Anomaly Scores`, `Python Logging Configuration`, `joblib (load/dump)` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Deep Learning Detector`** (2 nodes): `PyOD AutoEncoder`, `TensorFlow oneDNN/CUDA Warning`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PyCaret Anomaly Detection Pipeline Notebook` connect `Notebook Infrastructure` to `PyCaret Pipeline Core`, `Dataset & Contamination Estimation`, `Candidate Algorithm Pool`, `Results & Visualization`, `Deep Learning Detector`?**
  _High betweenness centrality (0.696) - this node is a cross-community bridge._
- **Why does `PyCaret Anomaly Detection Module` connect `PyCaret Pipeline Core` to `Dataset & Contamination Estimation`, `Notebook Infrastructure`, `Results & Visualization`?**
  _High betweenness centrality (0.530) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `PyOD IsolationForest (IForest)` (e.g. with `scikit-learn IsolationForest` and `PyOD Gaussian Mixture Model (GMM)`) actually correct?**
  _`PyOD IsolationForest (IForest)` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ROC AUC Score (sklearn.metrics)`, `Expense Category: Combustíveis e Lubrificantes`, `Plotly Scatter Plot of Anomaly Scores` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._