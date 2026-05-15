import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from datasets import load_dataset

TEXT_METRICS = ["TCC", "MS", "M"]
VIDEO_METRICS = ["VC", "VA", "VE"]
ALL_METRICS = TEXT_METRICS + VIDEO_METRICS


def _safe_corr(value):
    return 0.0 if np.isnan(value) else float(value)


def score_submission(submission_csv_path: str, split: str) -> dict:
    """
    Score a CSV submission.

    Expected CSV columns:
        id,TCC,MS,M,VC,VA,VE

    Expected score ranges:
        TCC, MS, M: 0, 1, 2
        VC, VA, VE: 0, 1, 2, 3
    """

    golden_truth = load_dataset(
        "analogy-evaluation/challenge-dataset",
        split=split,
    )

    submission = pd.read_csv(submission_csv_path)
    submission.columns = submission.columns.str.strip()

    required_cols = ["id"] + ALL_METRICS
    missing_cols = [col for col in required_cols if col not in submission.columns]

    if missing_cols:
        raise ValueError(
            f"Submission CSV is missing required columns: {missing_cols}. "
            f"Found columns: {list(submission.columns)}"
        )

    if len(submission) != len(golden_truth):
        raise ValueError(
            f"Submission has {len(submission)} rows, but gold split has "
            f"{len(golden_truth)} rows."
        )

    submission["id"] = submission["id"].astype(str)

    results = {"kendall": {}, "spearman": {}}
    ids = [str(i) for i in range(len(golden_truth))]

    # Make sure all expected ids are present.
    submitted_ids = set(submission["id"])
    missing_ids = [i for i in ids if i not in submitted_ids]

    if missing_ids:
        raise ValueError(f"Submission is missing ids: {missing_ids[:20]}")

    # Align predictions to gold order by id.
    submission = submission.set_index("id")

    for metric in ALL_METRICS:
        y_true = np.array(golden_truth[metric])
        y_pred = np.array([
            submission.loc[i, metric]
            for i in ids
        ])

        y_pred = pd.to_numeric(y_pred, errors="raise").astype(int)

        tau, _ = kendalltau(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)

        results["kendall"][metric] = _safe_corr(tau)
        results["spearman"][metric] = _safe_corr(rho)

    results["kendall"]["TEXT_AVG"] = float(
        np.mean([results["kendall"][m] for m in TEXT_METRICS])
    )
    results["kendall"]["VIDEO_AVG"] = float(
        np.mean([results["kendall"][m] for m in VIDEO_METRICS])
    )
    results["kendall"]["AVG"] = float(
        np.mean([results["kendall"][m] for m in ALL_METRICS])
    )

    results["spearman"]["TEXT_AVG"] = float(
        np.mean([results["spearman"][m] for m in TEXT_METRICS])
    )
    results["spearman"]["VIDEO_AVG"] = float(
        np.mean([results["spearman"][m] for m in VIDEO_METRICS])
    )
    results["spearman"]["AVG"] = float(
        np.mean([results["spearman"][m] for m in ALL_METRICS])
    )

    return results


def print_scores(results):
    print(f"{'metric':<10} {'tau-b':>8} {'rho':>8}")

    for metric in ALL_METRICS:
        print(
            f"{metric:<10} "
            f"{results['kendall'][metric]:>+8.3f} "
            f"{results['spearman'][metric]:>+8.3f}"
        )

    for metric in ["TEXT_AVG", "VIDEO_AVG", "AVG"]:
        print(
            f"{metric:<10} "
            f"{results['kendall'][metric]:>+8.3f} "
            f"{results['spearman'][metric]:>+8.3f}"
        )


if __name__ == "__main__":
    results = score_submission(
        submission_csv_path="submission.csv",
        split="validation",
    )
    print_scores(results)
