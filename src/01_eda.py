"""Exploratory data analysis for the diabetes health-indicators dataset.

Loads the raw CSV, deduplicates it, and produces summary statistics plus
five figures used in the paper. Writes results/eda_summary.txt and
figures/fig01..fig05.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
TARGET = "Diabetes_binary"

KEY_PREDICTORS = ["GenHlth", "HighBP", "BMI", "HighChol", "Age"]
RISK_FACTORS = [
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "DiffWalk",
    "HvyAlcoholConsump",
]


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    log("=" * 60)
    log("DIABETES HEALTH INDICATORS - EXPLORATORY DATA ANALYSIS")
    log("=" * 60)

    raw = pd.read_csv(DATA_PATH)
    log(f"\nRaw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")

    n_dupes = raw.duplicated().sum()
    log(f"Duplicate rows: {n_dupes}")

    df = raw.drop_duplicates().reset_index(drop=True)
    log(f"Shape after deduplication: {df.shape[0]} rows x {df.shape[1]} columns")

    n_missing = df.isnull().sum().sum()
    log(f"Missing values (total cells): {n_missing}")

    log("\n--- Class balance ---")
    balance = df[TARGET].value_counts(normalize=True).sort_index()
    for cls, pct in balance.items():
        label = "No diabetes" if cls == 0 else "Diabetes"
        log(f"  {label} ({int(cls)}): {pct:.4f} ({(df[TARGET] == cls).sum()} rows)")

    log("\n--- Descriptive statistics ---")
    log(df.describe().T.to_string())

    log("\n--- Correlation with target (sorted) ---")
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    log(corr.to_string())

    # Fig 1: class balance
    plt.figure(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax = sns.barplot(
        x=["No diabetes", "Diabetes"], y=counts.values, hue=["No diabetes", "Diabetes"],
        palette=["#4C72B0", "#C44E52"], legend=False,
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, str(v), ha="center", fontweight="bold")
    plt.title("Class balance: Diabetes_binary")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig01_class_balance.png"), dpi=150)
    plt.close()

    # Fig 2: correlation of each feature with the target
    plt.figure(figsize=(7, 8))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in corr.values]
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index, palette=colors, legend=False)
    plt.title("Feature correlation with Diabetes_binary")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig02_target_correlation.png"), dpi=150)
    plt.close()

    # Fig 3: full correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False,
        square=True, linewidths=0.3, cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig03_correlation_matrix.png"), dpi=150)
    plt.close()

    # Fig 4: key predictors vs target
    fig, axes = plt.subplots(1, len(KEY_PREDICTORS), figsize=(20, 4))
    for ax, col in zip(axes, KEY_PREDICTORS):
        sns.barplot(
            data=df, x=TARGET, y=col, hue=TARGET, palette=["#4C72B0", "#C44E52"],
            legend=False, ax=ax, errorbar=("ci", 95),
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No diabetes", "Diabetes"])
        ax.set_title(f"{col} vs target")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig04_key_predictors.png"), dpi=150)
    plt.close()

    # Fig 5: prevalence of binary risk factors by diabetes status
    plt.figure(figsize=(9, 5))
    prevalence = (
        df.groupby(TARGET)[RISK_FACTORS].mean().T.rename(
            columns={0: "No diabetes", 1: "Diabetes"}
        )
    )
    prevalence.plot(kind="bar", ax=plt.gca(), color=["#4C72B0", "#C44E52"])
    plt.title("Risk factor prevalence by diabetes status")
    plt.ylabel("Proportion = 1")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig05_risk_factors.png"), dpi=150)
    plt.close()

    log(f"\nSaved 5 figures to {FIG_DIR}")

    # Save deduplicated dataset for downstream scripts to reuse (keeps 02_models.py simple)
    dedup_path = os.path.join(BASE_DIR, "data", "diabetes_dedup.csv")
    df.to_csv(dedup_path, index=False)
    log(f"Saved deduplicated dataset to {dedup_path}")

    summary_path = os.path.join(RESULTS_DIR, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    log(f"\nSaved EDA summary to {summary_path}")


if __name__ == "__main__":
    main()
