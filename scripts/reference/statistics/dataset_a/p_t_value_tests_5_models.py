"""Paired statistical comparison of the five models using corresponding CV folds.

Primary analysis: proposed model versus each baseline using two-sided paired
t-tests on the five matching fold accuracies. Holm adjustment controls the
family-wise error rate across the four primary comparisons.

The script also saves all ten pairwise comparisons and a Friedman omnibus test.
No training or model checkpoints are required.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t, ttest_rel


ROOT = Path(r"C:\user\juli\thesis\original_file\othr folders")
OUTPUT_DIR = ROOT / "P_T_VALUE_TESTS_5_MODELS"
METRIC = "accuracy"
ALPHA = 0.05

MODEL_FILES = {
    "Proposed": (
        ROOT / "CV_RESULTS_PROPOSED_MODEL" / "PROPOSED_MODEL_5FOLD_RESULTS.csv"
    ),
    "VGG16": (
        ROOT / "CV_RESULTS_VGG16_RERUN" / "VGG16_5FOLD_RESULTS.csv"
    ),
    "ResNet50": (
        ROOT / "CV_RESULTS_RESNET50_SAFE_RERUN" / "RESNET50_5FOLD_RESULTS.csv"
    ),
    "EfficientNet-B3": (
        ROOT
        / "CV_RESULTS_EFFICIENTNET_B3_SAFE_RERUN"
        / "EFFICIENTNET_B3_5FOLD_RESULTS.csv"
    ),
    "DenseNet121": (
        ROOT
        / "CV_RESULTS_DENSENET121_SAFE_RERUN"
        / "DENSENET121_5FOLD_RESULTS.csv"
    ),
}


def load_fold_values(model_name: str, path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"{model_name} result file not found:\n{path}")

    frame = pd.read_csv(path)
    required = {"fold", METRIC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model_name} is missing columns: {sorted(missing)}")

    fold_numbers = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if len(frame) != 5 or set(fold_numbers) != {1, 2, 3, 4, 5}:
        raise ValueError(
            f"{model_name} must contain exactly one row for each fold 1-5."
        )
    if fold_numbers.duplicated().any():
        raise ValueError(f"{model_name} contains duplicate fold numbers.")

    values = pd.to_numeric(frame[METRIC], errors="raise").astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} contains a missing or invalid {METRIC}.")

    return pd.Series(values.to_numpy(), index=fold_numbers, name=model_name).sort_index()


def holm_adjust(raw_p_values):
    """Return Holm-adjusted p-values in the original comparison order."""
    p_values = np.asarray(raw_p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running_maximum = 0.0

    for rank, original_index in enumerate(order):
        adjusted = (len(p_values) - rank) * p_values[original_index]
        running_maximum = max(running_maximum, adjusted)
        adjusted_sorted[rank] = min(running_maximum, 1.0)

    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_test(model_a: str, model_b: str, table: pd.DataFrame) -> dict:
    a = table[model_a].to_numpy(dtype=float)
    b = table[model_b].to_numpy(dtype=float)
    differences = a - b
    test = ttest_rel(a, b, alternative="two-sided")
    difference_sd = float(np.std(differences, ddof=1))
    difference_se = difference_sd / np.sqrt(len(differences))
    critical_t = float(t.ppf(0.975, df=len(differences) - 1))
    mean_difference = float(np.mean(differences))
    cohens_dz = (
        float(mean_difference / difference_sd)
        if difference_sd > 0
        else np.nan
    )

    return {
        "Comparison": f"{model_a} vs {model_b}",
        "Model A": model_a,
        "Model B": model_b,
        "Mean accuracy A (%)": float(np.mean(a) * 100),
        "Mean accuracy B (%)": float(np.mean(b) * 100),
        "Mean difference A-B (percentage points)": float(mean_difference * 100),
        "95% CI lower (percentage points)": float(
            (mean_difference - critical_t * difference_se) * 100
        ),
        "95% CI upper (percentage points)": float(
            (mean_difference + critical_t * difference_se) * 100
        ),
        "t-value": float(test.statistic),
        "df": len(differences) - 1,
        "p-value": float(test.pvalue),
        "Cohen's dz": cohens_dz,
    }


def add_holm_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Holm-adjusted p"] = holm_adjust(frame["p-value"])
    frame["Significant?"] = np.where(frame["Holm-adjusted p"] < ALPHA, "Yes", "No")
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_table = pd.DataFrame(
        {name: load_fold_values(name, path) for name, path in MODEL_FILES.items()}
    )
    fold_table.index.name = "fold"
    fold_table.mul(100).reset_index().to_csv(
        OUTPUT_DIR / "FIVE_MODEL_FOLD_ACCURACIES.csv", index=False
    )

    # Model-specific uncertainty across the five matching folds. Values are
    # reported as percentages; the 95% CI uses Student's t with df = 4.
    critical_t = float(t.ppf(0.975, df=len(fold_table) - 1))
    model_summary_rows = []
    for model_name in MODEL_FILES:
        values_percent = fold_table[model_name].to_numpy(dtype=float) * 100
        mean_value = float(np.mean(values_percent))
        sd_value = float(np.std(values_percent, ddof=1))
        se_value = float(sd_value / np.sqrt(len(values_percent)))
        margin = critical_t * se_value
        model_summary_rows.append(
            {
                "Model": model_name,
                "Mean accuracy (%)": mean_value,
                "SD (%)": sd_value,
                "SE (%)": se_value,
                "95% CI lower (%)": mean_value - margin,
                "95% CI upper (%)": mean_value + margin,
                "Folds": len(values_percent),
            }
        )
    pd.DataFrame(model_summary_rows).to_csv(
        OUTPUT_DIR / "MODEL_WISE_ACCURACY_95_PERCENT_CONFIDENCE_INTERVALS.csv",
        index=False,
    )

    baselines = [name for name in MODEL_FILES if name != "Proposed"]
    primary = pd.DataFrame(
        [paired_test("Proposed", baseline, fold_table) for baseline in baselines]
    )
    primary = add_holm_results(primary)
    primary.to_csv(
        OUTPUT_DIR / "PROPOSED_VS_BASELINES_PAIRED_T_TESTS.csv", index=False
    )

    all_pairwise = pd.DataFrame(
        [paired_test(a, b, fold_table) for a, b in combinations(MODEL_FILES, 2)]
    )
    all_pairwise = add_holm_results(all_pairwise)
    all_pairwise.to_csv(
        OUTPUT_DIR / "ALL_5_MODELS_PAIRWISE_PAIRED_T_TESTS.csv", index=False
    )

    friedman = friedmanchisquare(
        *(fold_table[name].to_numpy(dtype=float) for name in MODEL_FILES)
    )
    pd.DataFrame(
        [
            {
                "test": "Friedman omnibus test",
                "metric": METRIC,
                "models": len(MODEL_FILES),
                "matched_folds": len(fold_table),
                "chi_square": float(friedman.statistic),
                "degrees_of_freedom": len(MODEL_FILES) - 1,
                "p_value": float(friedman.pvalue),
                "Significant?": "Yes" if friedman.pvalue < ALPHA else "No",
            }
        ]
    ).to_csv(OUTPUT_DIR / "FRIEDMAN_OMNIBUS_TEST.csv", index=False)

    print("\nFold accuracies used:\n")
    print((fold_table * 100).round(6).to_string())
    print("\nProposed model versus baselines:\n")
    print(
        primary[
            [
                "Comparison",
                "t-value",
                "p-value",
                "Holm-adjusted p",
                "Significant?",
            ]
        ].to_string(index=False)
    )
    print(f"\nAll statistical outputs saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
