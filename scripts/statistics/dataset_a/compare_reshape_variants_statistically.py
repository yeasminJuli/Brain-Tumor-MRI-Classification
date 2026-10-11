"""Statistical comparison of row-, column-, and alternative-wise 5-fold CV.

This script reads existing fold results only. It does not train a model and does
not alter the manuscript or source result CSV files.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"C:\user\juli\thesis\original_file")
INPUTS = {
    "Row-wise": ROOT / "othr folders" / "CV_RESULTS_PROPOSED_MODEL" / "PROPOSED_MODEL_5FOLD_RESULTS.csv",
    "Column-wise": ROOT / "othr folders" / "CV_RESULTS_COLUMN_WISE" / "COLUMN_WISE_5FOLD_RESULTS.csv",
    "Alternative-wise": ROOT / "othr folders" / "CV_RESULTS_ALTERNATIVE_WISE" / "ALTERNATIVE_WISE_5FOLD_RESULTS.csv",
}
OUTPUT_DIR = ROOT / "othr folders" / "RESHAPING_ABLATION_STATISTICAL_COMPARISON"

METRIC_ALIASES = {
    "Accuracy": ("accuracy",),
    "Macro Precision": ("macro_precision", "precision_macro"),
    "Macro Recall": ("macro_recall", "recall_macro"),
    "Macro F1": ("macro_f1", "f1_macro"),
    "Macro AUC": ("macro_auc", "auc_macro"),
}


def metric_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    for name in aliases:
        if name in frame.columns:
            return name
    raise KeyError(f"None of these columns exists: {aliases}")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for ordering, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        frame = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
        if frame["fold"].tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"{ordering} does not contain folds 1-5 exactly once.")
        if "validation_samples" in frame and not (frame["validation_samples"] == 5712).all():
            raise ValueError(f"Unexpected validation sample count in {ordering}.")
        frames[ordering] = frame

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_critical = stats.t.ppf(0.975, df=4)

    # 1. Fold-level accuracy and classification error rate.
    fold_rows = []
    for fold in range(1, 6):
        row = {"Fold": fold}
        for ordering, frame in frames.items():
            acc = float(frame.loc[frame["fold"] == fold, "accuracy"].iloc[0]) * 100
            row[f"{ordering} Accuracy (%)"] = acc
            row[f"{ordering} Error Rate (%)"] = 100.0 - acc
        fold_rows.append(row)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(OUTPUT_DIR / "01_RESHAPING_ABLATION_FOLD_ACCURACY_AND_ERROR_RATE.csv", index=False)

    # 2. Across-fold variability for every common metric.
    summary_rows = []
    for ordering, frame in frames.items():
        for metric, aliases in METRIC_ALIASES.items():
            col = metric_column(frame, aliases)
            values = frame[col].to_numpy(dtype=float) * 100
            mean = float(np.mean(values))
            variance = float(np.var(values, ddof=1))
            sd = math.sqrt(variance)
            se = sd / math.sqrt(len(values))
            margin = t_critical * se
            summary_rows.append({
                "Ordering": ordering,
                "Metric": metric,
                "Mean (%)": mean,
                "Sample Variance (%^2)": variance,
                "SD (%)": sd,
                "SE (%)": se,
                "95% CI Lower (%)": mean - margin,
                "95% CI Upper (%)": mean + margin,
                "N Folds": len(values),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "02_RESHAPING_ABLATION_METRIC_VARIANCE_SD_SE_95CI.csv", index=False)

    # 3. Accuracy paired t-tests because the same five folds were used.
    comparisons = [
        ("Row-wise", "Column-wise"),
        ("Row-wise", "Alternative-wise"),
        ("Column-wise", "Alternative-wise"),
    ]
    test_rows = []
    raw_p = []
    for first, second in comparisons:
        a = frames[first]["accuracy"].to_numpy(dtype=float) * 100
        b = frames[second]["accuracy"].to_numpy(dtype=float) * 100
        difference = a - b
        mean_difference = float(np.mean(difference))
        sd_difference = float(np.std(difference, ddof=1))
        se_difference = sd_difference / math.sqrt(len(difference))
        margin = t_critical * se_difference
        t_value, p_value = stats.ttest_rel(a, b)
        dz = mean_difference / sd_difference if sd_difference else np.nan
        raw_p.append(float(p_value))
        test_rows.append({
            "Comparison": f"{first} vs {second}",
            "Difference Direction": f"{first} minus {second}",
            "Mean Accuracy Difference (percentage points)": mean_difference,
            "Difference SD (percentage points)": sd_difference,
            "Difference SE (percentage points)": se_difference,
            "Difference 95% CI Lower (percentage points)": mean_difference - margin,
            "Difference 95% CI Upper (percentage points)": mean_difference + margin,
            "Paired t-value": float(t_value),
            "Degrees of Freedom": 4,
            "Raw p-value": float(p_value),
            "Cohen dz": dz,
        })
    adjusted = holm_adjust(raw_p)
    for row, p_adjusted in zip(test_rows, adjusted):
        row["Holm-adjusted p-value"] = p_adjusted
        row["Significant after Holm (alpha=0.05)"] = "Yes" if p_adjusted < 0.05 else "No"
    paired_tests = pd.DataFrame(test_rows)
    paired_tests.to_csv(OUTPUT_DIR / "03_RESHAPING_ABLATION_PAIRED_T_TESTS_HOLM.csv", index=False)

    # 4. Omnibus repeated-measures comparison across all three orderings.
    arrays = [frames[name]["accuracy"].to_numpy(dtype=float) * 100 for name in INPUTS]
    friedman_stat, friedman_p = stats.friedmanchisquare(*arrays)
    omnibus = pd.DataFrame([{
        "Test": "Friedman test across three ordering schemes",
        "Statistic": float(friedman_stat),
        "Degrees of Freedom": 2,
        "p-value": float(friedman_p),
        "Significant (alpha=0.05)": "Yes" if friedman_p < 0.05 else "No",
        "N Matched Folds": 5,
        "Interpretation": "At least one ordering differs" if friedman_p < 0.05 else "No overall difference detected",
    }])
    omnibus.to_csv(OUTPUT_DIR / "04_RESHAPING_ABLATION_FRIEDMAN_OMNIBUS_TEST.csv", index=False)

    # 5. Compact manuscript-ready accuracy table.
    compact_rows = []
    for ordering, frame in frames.items():
        values = frame["accuracy"].to_numpy(dtype=float) * 100
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1))
        sd = math.sqrt(variance)
        se = sd / math.sqrt(5)
        margin = t_critical * se
        compact_rows.append({
            "Ordering Scheme": ordering,
            "Mean Accuracy (%)": mean,
            "Classification Error Rate (%)": 100.0 - mean,
            "Variance (%^2)": variance,
            "SD (%)": sd,
            "SE (%)": se,
            "95% CI Lower (%)": mean - margin,
            "95% CI Upper (%)": mean + margin,
            "Mean +/- SD (%)": f"{mean:.3f} +/- {sd:.3f}",
            "95% CI (%)": f"{mean - margin:.3f}-{mean + margin:.3f}",
        })
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(OUTPUT_DIR / "05_RESHAPING_ABLATION_MANUSCRIPT_READY_ACCURACY_TABLE.csv", index=False)

    print(f"Completed. Results saved to: {OUTPUT_DIR}")
    print("\nManuscript-ready accuracy summary:")
    print(compact.to_string(index=False))
    print("\nPaired accuracy comparisons:")
    print(paired_tests.to_string(index=False))
    print("\nOmnibus test:")
    print(omnibus.to_string(index=False))


if __name__ == "__main__":
    main()
