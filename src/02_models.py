"""Train, tune and evaluate four classifiers on the diabetes dataset.

Pipeline (per spec): StandardScaler refit inside every CV fold -> no leakage.
5-fold stratified GridSearchCV optimising ROC-AUC -> 5-fold CV mean +/- std on
the training set -> a single final evaluation on the held-out test set.
Saves every trained pipeline plus the best-performing one as models/best_model.pkl,
and writes all result tables under results/.
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

TARGET = "Diabetes_binary"
RANDOM_STATE = 42
CV_FOLDS = 5

MODEL_SPECS = {
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [4, 6, 8, 10, None],
            "clf__min_samples_leaf": [1, 20, 50],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_leaf": [1, 20],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
        },
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "clf__n_neighbors": [5, 15, 25, 51],
            "clf__weights": ["uniform", "distance"],
        },
    },
}

DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
}

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def log(lines, msg=""):
    print(msg)
    lines.append(str(msg))


def main():
    lines = []
    t0 = time.time()

    log(lines, "=" * 60)
    log(lines, "DIABETES PREDICTION - MODEL TRAINING & EVALUATION")
    log(lines, "=" * 60)

    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET].astype(int)

    log(lines, f"\nDataset: {df.shape[0]} rows, {len(features)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log(lines, f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows (untouched until final eval)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    tuning_rows = []
    crossval_rows = []
    comparison_rows = []
    fitted = {}
    test_predictions = {}  # name -> (y_pred, y_proba), for ROC curves & confusion matrices

    for key, spec in MODEL_SPECS.items():
        name = DISPLAY_NAMES[key]
        log(lines, f"\n{'-' * 60}\n{name}\n{'-' * 60}")

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", spec["estimator"])])

        t_start = time.time()
        grid = GridSearchCV(
            pipe, spec["param_grid"], scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        elapsed = time.time() - t_start

        log(lines, f"Best params: {grid.best_params_}")
        log(lines, f"Best CV ROC-AUC (from GridSearchCV): {grid.best_score_:.4f}")
        log(lines, f"Grid search time: {elapsed:.1f}s")

        for params, mean_score, std_score in zip(
            grid.cv_results_["params"],
            grid.cv_results_["mean_test_score"],
            grid.cv_results_["std_test_score"],
        ):
            tuning_rows.append(
                {"model": name, "params": json.dumps(params), "mean_roc_auc": mean_score, "std_roc_auc": std_score}
            )

        best_model = grid.best_estimator_
        fitted[key] = best_model

        # 5-fold CV on the training set with the tuned pipeline -> mean +/- std per metric
        cv_res = cross_validate(best_model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        log(lines, "5-fold CV on training set (tuned model):")
        cv_row = {"model": name}
        for metric in SCORING:
            scores = cv_res[f"test_{metric}"]
            cv_row[f"{metric}_mean"] = scores.mean()
            cv_row[f"{metric}_std"] = scores.std()
            log(lines, f"  {metric:10s}: {scores.mean():.4f} +/- {scores.std():.4f}")
        crossval_rows.append(cv_row)

        # Single final evaluation on the held-out test set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        log(lines, "Test set evaluation:")
        log(lines, f"  Accuracy : {acc:.4f}")
        log(lines, f"  Precision: {prec:.4f}")
        log(lines, f"  Recall   : {rec:.4f}")
        log(lines, f"  F1-score : {f1:.4f}")
        log(lines, f"  ROC-AUC  : {auc:.4f}")
        log(lines, f"  Confusion matrix [[TN FP] [FN TP]]:\n{cm}")

        comparison_rows.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
                "tn": cm[0, 0],
                "fp": cm[0, 1],
                "fn": cm[1, 0],
                "tp": cm[1, 1],
            }
        )

        test_predictions[name] = (y_pred, y_proba)
        joblib.dump(best_model, os.path.join(MODELS_DIR, f"{key}.pkl"))

    # --- Pick overall best model by test-set ROC-AUC ---
    comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
    best_row = comparison_df.iloc[0]
    best_key = [k for k, v in DISPLAY_NAMES.items() if v == best_row["model"]][0]
    best_model = fitted[best_key]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

    log(lines, f"\n{'=' * 60}")
    log(lines, f"BEST MODEL (by test ROC-AUC): {best_row['model']} (ROC-AUC = {best_row['roc_auc']:.4f})")
    log(lines, "=" * 60)

    # --- Fig 6: model comparison (bar chart of test-set metrics) ---
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_df = comparison_df.set_index("model")[metrics]
    plot_df.plot(kind="bar", figsize=(10, 5), colormap="tab10")
    plt.title("Model comparison - test set metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig06_model_comparison.png"), dpi=150)
    plt.close()

    # --- Fig 7: ROC curves for all four models ---
    plt.figure(figsize=(7, 6))
    for name, (_, y_proba) in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves - test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig07_roc_curves.png"), dpi=150)
    plt.close()

    # --- Fig 8: confusion matrices, one subplot per model ---
    fig, axes = plt.subplots(1, len(test_predictions), figsize=(5 * len(test_predictions), 4.5))
    for ax, (name, (y_pred, _)) in zip(axes, test_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig08_confusion_matrices.png"), dpi=150)
    plt.close()

    # --- Interpretability ---
    # Random Forest: Gini feature importance
    rf_model = fitted["random_forest"].named_steps["clf"]
    fi_df = (
        pd.DataFrame({"feature": features, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance_rf.csv"), index=False)
    log(lines, "\nRandom Forest - top 5 features by Gini importance:")
    log(lines, fi_df.head(5).to_string(index=False))

    # Logistic Regression: standardised coefficients (StandardScaler is inside the pipeline)
    logreg_model = fitted["logistic_regression"].named_steps["clf"]
    coef_df = (
        pd.DataFrame({"feature": features, "coefficient": logreg_model.coef_[0]})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    coef_df.to_csv(os.path.join(RESULTS_DIR, "coefficients_logreg.csv"), index=False)
    log(lines, "\nLogistic Regression - top 5 standardised coefficients:")
    log(lines, coef_df.head(5).to_string(index=False))

    # Permutation importance (model-agnostic) on the overall best model, test set
    log(lines, f"\nComputing permutation importance for best model ({best_row['model']}) on test set...")
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
    )
    perm_df = (
        pd.DataFrame(
            {"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm_df.to_csv(os.path.join(RESULTS_DIR, "permutation_importance.csv"), index=False)
    log(lines, "Permutation importance - top 5 features:")
    log(lines, perm_df.head(5).to_string(index=False))

    # --- Fig 9: feature importance - RF Gini, LogReg |coef|, permutation importance ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    top_n = 12

    rf_top = fi_df.head(top_n)
    sns.barplot(data=rf_top, x="importance", y="feature", hue="feature", palette="Blues_r", legend=False, ax=axes[0])
    axes[0].set_title("Random Forest - Gini importance")

    coef_top = coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index).head(top_n)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in coef_top["coefficient"]]
    sns.barplot(data=coef_top, x="coefficient", y="feature", hue="feature", palette=colors, legend=False, ax=axes[1])
    axes[1].set_title("Logistic Regression - standardised coefficients")

    perm_top = perm_df.head(top_n)
    sns.barplot(data=perm_top, x="importance_mean", y="feature", hue="feature", palette="Greens_r", legend=False, ax=axes[2])
    axes[2].set_title(f"Permutation importance ({best_row['model']})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig09_feature_importance.png"), dpi=150)
    plt.close()
    log(lines, f"\nSaved fig06-fig09 to {FIG_DIR}")

    # --- Save result tables ---
    pd.DataFrame(tuning_rows).to_csv(os.path.join(RESULTS_DIR, "tuning_results.csv"), index=False)
    pd.DataFrame(crossval_rows).to_csv(os.path.join(RESULTS_DIR, "crossval_scores.csv"), index=False)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison_test_set.csv"), index=False)

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(features, f, indent=2)

    total_elapsed = time.time() - t0
    log(lines, f"\nTotal run time: {total_elapsed / 60:.1f} min")

    with open(os.path.join(RESULTS_DIR, "model_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved model_report.txt and all result CSVs to {RESULTS_DIR}")
    print(f"Saved trained pipelines and best_model.pkl to {MODELS_DIR}")


if __name__ == "__main__":
    main()
