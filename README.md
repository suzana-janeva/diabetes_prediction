# Diabetes Prediction

Predicting diabetes from health indicators using four classification algorithms:
Decision Tree
Random Forest
Logistic Regression 
K-Nearest Neighbors 

## Data

(Kaggle: `alexteboul/diabetes-health-indicators-dataset`),

## Setup

```bash
pip3 install --user -r requirements.txt
```

## Run the pipeline

```bash
python3 src/01_eda.py      # exploration -> figures/fig01-05, results/eda_summary.txt
python3 src/02_models.py   # train, tune, evaluate, save -> models/*.pkl, figures/fig06-09, results/*.csv
python3 -m streamlit run src/03_app.py   # launch the web UI
```

Streamlit prints a local URL (default `http://localhost:8501`) — open it in a browser.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.7459 | 0.7283 | 0.7977 | 0.7614 | **0.8199** |
| Logistic Regression | 0.7446 | 0.7374 | 0.7726 | 0.7546 | 0.8174 |
| Decision Tree | 0.7340 | 0.7234 | 0.7717 | 0.7468 | 0.8093 |
| K-Nearest Neighbors | 0.7359 | 0.7179 | 0.7912 | 0.7528 | 0.8074 |

Random Forest is the best performer by ROC-AUC and is saved as `models/best_model.pkl`. On medical data, recall matters most (a false negative means missing a diabetic patient); KNN and Random Forest lead on recall.

Top predictors (by correlation, RF Gini importance, and permutation importance): `GenHlth`, `HighBP`, `BMI`, `HighChol`, `Age`.

## Project structure

```
diabetes-prediction/
├── data/                  raw + deduplicated CSV
├── src/
│   ├── 01_eda.py          exploration + 5 figures
│   ├── 02_models.py       train, tune, evaluate, save .pkl, 4 more figures
│   └── 03_app.py          Streamlit UI
├── models/                best_model.pkl + one .pkl per algorithm
├── figures/                fig01-09
├── results/                eda_summary, model_report, tuning/CV/comparison/importance CSVs
├── requirements.txt
└── README.md
```
