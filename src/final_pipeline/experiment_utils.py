from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from final_pipeline.common import PROJECT_ROOT, write_json


QEVASION_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
QEVASION_LABEL_ZH = {
    "Clear Reply": "明确回答",
    "Ambivalent": "含糊/矛盾回答",
    "Clear Non-Reply": "明确不回答",
}
QNLI_LABELS = ["not_entailment", "entailment"]
QNLI_LABEL_ZH = {
    "not_entailment": "不含答案",
    "entailment": "包含答案",
}


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def metric_dict(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str],
    label_names: list[str],
    path: Path,
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    write_json(path, report)
    return report


def save_confusion_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str],
    display_labels: list[str],
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7.4, 6.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def probability_columns(classes: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        probabilities,
        columns=[f"prob_{label}" for label in classes],
    )


def save_predictions(
    frame: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    prediction = frame.copy()
    prediction["gold"] = list(y_true)
    prediction["prediction"] = list(y_pred)
    prediction["is_error"] = prediction["gold"].ne(prediction["prediction"])
    prediction = pd.concat(
        [
            prediction.reset_index(drop=True),
            probability_columns(classes, probabilities).reset_index(drop=True),
        ],
        axis=1,
    )
    prediction["confidence"] = probabilities.max(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False, encoding="utf-8-sig")
    return prediction


def run_metadata() -> dict[str, Any]:
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "started_at_unix": time.time(),
    }


def finalize_run_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    metadata["finished_at_unix"] = time.time()
    metadata["elapsed_seconds"] = round(
        metadata["finished_at_unix"] - metadata["started_at_unix"],
        3,
    )
    write_json(path, metadata)
    return metadata


def paths_for_step(step_name: str) -> dict[str, Path]:
    root = PROJECT_ROOT / "results" / step_name
    result = {
        "root": root,
        "figures": root / "figures",
        "predictions": root / "predictions",
        "models": root / "models",
        "reports": root / "classification_reports",
        "features": root / "features",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def log_dataframe(frame: pd.DataFrame, title: str) -> None:
    logging.info("%s\n%s", title, frame.to_string(index=False))
