"""
Evaluation Metrics Module.

Computes Accuracy, Macro-F1, and Matthews Correlation Coefficient (MCC)
as defined in Section 6.1.1.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


LABEL_TO_INT = {
    "Class 1 (Low Value)": 0,
    "Class 2 (Medium Value)": 1,
    "Class 3 (High Value)": 2,
}


def evaluate_predictions(predictions_path, print_report: bool = True) -> dict:
    """Evaluate model predictions against ground truth.

    Args:
        predictions_path: Path to JSONL file with predictions
                          (must contain 'predicted_label' and 'ground_truth').
        print_report: Whether to print the full classification report.

    Returns:
        Dictionary with accuracy, macro_f1, and mcc scores.
    """
    predictions_path = Path(predictions_path)

    y_true = []
    y_pred = []
    failed = 0

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            gt = record.get("ground_truth", "")
            pred = record.get("predicted_label", "")

            if gt not in LABEL_TO_INT:
                continue

            if pred not in LABEL_TO_INT:
                failed += 1
                y_pred.append(-1)
            else:
                y_pred.append(LABEL_TO_INT[pred])

            y_true.append(LABEL_TO_INT[gt])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle failed parses by assigning majority class
    if failed > 0:
        majority = np.bincount(y_true).argmax()
        y_pred[y_pred == -1] = majority
        print(f"Warning: {failed} predictions could not be parsed. Assigned to majority class.")

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)

    results = {
        "accuracy": round(accuracy * 100, 1),
        "macro_f1": round(macro_f1 * 100, 1),
        "mcc": round(mcc, 4),
        "total_samples": len(y_true),
        "parse_failures": failed,
    }

    if print_report:
        print("=" * 60)
        print("ERA Evaluation Results")
        print("=" * 60)
        print(f"  Accuracy:  {results['accuracy']}%")
        print(f"  Macro-F1:  {results['macro_f1']}%")
        print(f"  MCC:       {results['mcc']}")
        print(f"  Samples:   {results['total_samples']}")
        print(f"  Parse failures: {results['parse_failures']}")
        print("-" * 60)
        print("\nClassification Report:")
        target_names = ["Class 1 (Low)", "Class 2 (Medium)", "Class 3 (High)"]
        print(classification_report(y_true, y_pred, target_names=target_names))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ERA predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL")
    args = parser.parse_args()
    evaluate_predictions(args.predictions)
