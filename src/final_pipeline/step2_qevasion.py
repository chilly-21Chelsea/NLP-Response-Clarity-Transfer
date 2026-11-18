from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from final_pipeline.common import PROJECT_ROOT, configure_logging, load_json, write_json
from final_pipeline.experiment_utils import (
    QEVASION_LABEL_ZH,
    QEVASION_LABELS,
    configure_plots,
    finalize_run_metadata,
    log_dataframe,
    metric_dict,
    paths_for_step,
    run_metadata,
    save_classification_report,
    save_confusion_plot,
    save_predictions,
)


LABEL_ORDER = QEVASION_LABELS
LABEL_DISPLAY = [f"{label} / {QEVASION_LABEL_ZH[label]}" for label in LABEL_ORDER]
TRANSITIONS = ("but", "however", "although", "still", "though")
REFUSALS = (
    "no comment",
    "won't",
    "will not",
    "cannot",
    "can't",
    "decline",
    "not going to",
    "i don't know",
    "i do not know",
)
HEDGES = ("perhaps", "maybe", "i think", "we'll see", "let me", "as i said")


def pair_text(frame: pd.DataFrame) -> pd.Series:
    return frame["text_a"].fillna("") + " [SEP] " + frame["text_b"].fillna("")


def build_tfidf(ngram_range: tuple[int, int], class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    class_weight=class_weight,
                    max_iter=4000,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def save_evasion_eda(train: pd.DataFrame, paths: dict[str, Path]) -> None:
    counts = train["label"].value_counts().reindex(LABEL_ORDER)
    plt.figure(figsize=(8.0, 4.8))
    counts.plot(kind="bar", color="#4C78A8")
    plt.title("QEvasion clarity label distribution")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "qevasion_clarity_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    evasion_counts = train.loc[train["evasion_label"].ne(""), "evasion_label"].value_counts()
    plt.figure(figsize=(9.2, 5.2))
    evasion_counts.plot(kind="bar", color="#E09F3E")
    plt.title("QEvasion fine-grained evasion label distribution")
    plt.xlabel("evasion_label")
    plt.ylabel("count")
    plt.xticks(rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "qevasion_evasion_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    length_frame = train[["label", "text_a", "text_b"]].copy()
    length_frame["question_chars"] = length_frame["text_a"].str.len()
    length_frame["answer_chars"] = length_frame["text_b"].str.len()
    plt.figure(figsize=(9.0, 5.0))
    sns.boxplot(
        data=length_frame,
        x="label",
        y="answer_chars",
        order=LABEL_ORDER,
        showfliers=False,
    )
    plt.title("QEvasion answer length by clarity label")
    plt.xlabel("clarity label")
    plt.ylabel("characters")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "qevasion_length_by_label.png", dpi=180, bbox_inches="tight")
    plt.close()


def run_tfidf_suite(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_text: pd.Series,
    test_text: pd.Series,
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Pipeline], pd.DataFrame]:
    majority_label = train["label"].value_counts().index[0]
    majority_pred = np.repeat(majority_label, len(test))
    majority = metric_dict(test["label"], majority_pred, LABEL_ORDER)
    majority.update(
        {
            "model": "majority_baseline",
            "ngram": "n/a",
            "class_weight": "n/a",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
    )

    configs = [
        ("tfidf_uni_lr_none", "unigram", (1, 1), None),
        ("tfidf_uni_lr_balanced", "unigram", (1, 1), "balanced"),
        ("tfidf_uni_bi_lr_none", "unigram+bigram", (1, 2), None),
        ("tfidf_uni_bi_lr_balanced", "unigram+bigram", (1, 2), "balanced"),
    ]
    results = [majority]
    models: dict[str, Pipeline] = {}
    prediction_rows: list[pd.DataFrame] = []

    majority_probs = np.zeros((len(test), len(LABEL_ORDER)))
    majority_probs[:, LABEL_ORDER.index(majority_label)] = 1.0
    prediction_rows.append(
        save_predictions(
            test,
            test["label"],
            majority_pred,
            majority_probs,
            np.asarray(LABEL_ORDER),
            paths["predictions"] / "predictions_majority_baseline.csv",
        )
    )

    for name, ngram_name, ngram_range, class_weight in configs:
        logging.info("Training %s", name)
        model = build_tfidf(ngram_range, class_weight)
        model.fit(train_text, train["label"])
        predicted = model.predict(test_text)
        probabilities = model.predict_proba(test_text)
        metrics = metric_dict(test["label"], predicted, LABEL_ORDER)
        metrics.update(
            {
                "model": name,
                "ngram": ngram_name,
                "class_weight": class_weight or "none",
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        )
        results.append(metrics)
        models[name] = model
        joblib.dump(model, paths["models"] / f"{name}.joblib")
        save_classification_report(
            test["label"],
            predicted,
            LABEL_ORDER,
            LABEL_DISPLAY,
            paths["reports"] / f"{name}.json",
        )
        prediction_rows.append(
            save_predictions(
                test,
                test["label"],
                predicted,
                probabilities,
                model.classes_,
                paths["predictions"] / f"predictions_{name}.csv",
            )
        )

    return results, models, pd.concat(prediction_rows, ignore_index=True)


def grouped_cross_validation(
    train: pd.DataFrame,
    train_text: pd.Series,
    model_name: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    if model_name == "tfidf_uni_bi_lr_balanced":
        model = build_tfidf((1, 2), "balanced")
    elif model_name == "tfidf_uni_bi_lr_none":
        model = build_tfidf((1, 2), None)
    elif model_name == "tfidf_uni_lr_balanced":
        model = build_tfidf((1, 1), "balanced")
    else:
        model = build_tfidf((1, 1), None)
    groups = train["group"].replace("", "unknown")
    folds = GroupKFold(n_splits=5)
    scores = cross_val_score(
        model,
        train_text,
        train["label"],
        cv=folds,
        groups=groups,
        scoring="f1_macro",
        n_jobs=1,
    )
    result = {
        "model": model_name,
        "splitter": "GroupKFold_by_interview_url",
        "folds": [float(value) for value in scores],
        "mean_macro_f1": float(np.mean(scores)),
        "std_macro_f1": float(np.std(scores)),
        "groups": int(groups.nunique()),
    }
    write_json(paths["root"] / "grouped_cross_validation.json", result)
    return result


def evaluate_group_clean_test(
    best_model: Pipeline,
    best_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_text: pd.Series,
    paths: dict[str, Path],
) -> dict[str, Any]:
    train_groups = set(train["group"].dropna()) - {""}
    mask = ~test["group"].isin(train_groups)
    clean_test = test.loc[mask].reset_index(drop=True)
    clean_text = test_text.loc[mask].reset_index(drop=True)
    predicted = best_model.predict(clean_text)
    probabilities = best_model.predict_proba(clean_text)
    metrics = metric_dict(clean_test["label"], predicted, LABEL_ORDER)
    metrics.update(
        {
            "model": best_name,
            "remaining_rows": int(mask.sum()),
            "removed_rows": int((~mask).sum()),
        }
    )
    save_classification_report(
        clean_test["label"],
        predicted,
        LABEL_ORDER,
        LABEL_DISPLAY,
        paths["reports"] / f"{best_name}_group_clean.json",
    )
    save_predictions(
        clean_test,
        clean_test["label"],
        predicted,
        probabilities,
        best_model.classes_,
        paths["predictions"] / f"predictions_{best_name}_group_clean.csv",
    )
    write_json(paths["root"] / "group_clean_metrics.json", metrics)
    return metrics


def evaluate_group_clean_feature_model(
    best_model: LogisticRegression,
    best_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_features: np.ndarray,
    paths: dict[str, Path],
) -> dict[str, Any]:
    train_groups = set(train["group"].dropna()) - {""}
    mask = ~test["group"].isin(train_groups)
    clean_test = test.loc[mask].reset_index(drop=True)
    clean_features = test_features[mask.to_numpy()]
    predicted = best_model.predict(clean_features)
    probabilities = best_model.predict_proba(clean_features)
    metrics = metric_dict(clean_test["label"], predicted, LABEL_ORDER)
    metrics.update(
        {
            "model": best_name,
            "remaining_rows": int(mask.sum()),
            "removed_rows": int((~mask).sum()),
        }
    )
    save_classification_report(
        clean_test["label"],
        predicted,
        LABEL_ORDER,
        LABEL_DISPLAY,
        paths["reports"] / f"{best_name}_group_clean.json",
    )
    save_predictions(
        clean_test,
        clean_test["label"],
        predicted,
        probabilities,
        best_model.classes_,
        paths["predictions"] / f"predictions_{best_name}_group_clean.csv",
    )
    write_json(paths["root"] / "group_clean_minilm_metrics.json", metrics)
    return metrics


def answer_chunks(text: str, n_chunks: int = 3, chunk_chars: int = 1200) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    if not text:
        return [""] * n_chunks
    if len(text) <= chunk_chars:
        return [text] * n_chunks
    starts = [
        0,
        max(0, (len(text) - chunk_chars) // 2),
        max(0, len(text) - chunk_chars),
    ]
    return [text[start : start + chunk_chars] for start in starts[:n_chunks]]


def engineered_text_features(frame: pd.DataFrame) -> np.ndarray:
    question = frame["text_a"].fillna("").astype(str)
    answer = frame["text_b"].fillna("").astype(str)
    lower_answer = answer.str.lower()
    q_len = question.str.len().to_numpy()
    a_len = answer.str.len().to_numpy()
    q_words = question.str.split().str.len().to_numpy()
    a_words = answer.str.split().str.len().to_numpy()
    sentences = answer.str.count(r"[.!?]+").to_numpy() + 1
    question_marks = question.str.count(r"\?").to_numpy()
    transitions = np.array(
        [sum(text.count(word) for word in TRANSITIONS) for text in lower_answer],
        dtype=float,
    )
    refusals = np.array(
        [sum(text.count(word) for word in REFUSALS) for text in lower_answer],
        dtype=float,
    )
    hedges = np.array(
        [sum(text.count(word) for word in HEDGES) for text in lower_answer],
        dtype=float,
    )
    first_sentence = answer.str.split(r"(?<=[.!?])\s+", regex=True).str[0].str.len().to_numpy()
    return np.column_stack(
        [
            np.log1p(q_len),
            np.log1p(a_len),
            np.log1p(q_words),
            np.log1p(a_words),
            np.log1p(sentences),
            np.log1p(question_marks),
            np.log1p(transitions),
            np.log1p(refusals),
            np.log1p(hedges),
            np.log1p(first_sentence),
            np.log1p(a_len / np.maximum(q_len, 1)),
        ]
    ).astype(np.float32)


def load_sentence_encoder(model_dir: Path):
    from sentence_transformers import SentenceTransformer

    logging.info("Loading MiniLM from %s", model_dir)
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(model_dir),
        device="cpu",
        local_files_only=model_dir.exists(),
    )


def encode_baseline_features(
    frame: pd.DataFrame,
    encoder,
    output_path: Path,
) -> np.ndarray:
    if output_path.exists():
        logging.info("Using cached MiniLM baseline features: %s", output_path)
        return np.load(output_path)
    texts = (
        frame["text_a"].fillna("")
        + " [SEP] "
        + frame["text_b"].fillna("").map(
            lambda value: value
            if len(str(value)) <= 2200
            else str(value)[:1400] + " [TRUNCATED] " + str(value)[-600:]
        )
    )
    features = encoder.encode(
        texts.tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    return features


def encode_improved_features(
    frame: pd.DataFrame,
    encoder,
    scaler: StandardScaler | None,
    interaction_path: Path,
    numeric_path: Path,
    scaler_path: Path,
) -> tuple[np.ndarray, StandardScaler]:
    if interaction_path.exists() and numeric_path.exists() and scaler_path.exists():
        logging.info("Using cached MiniLM improved features: %s", interaction_path)
        fitted_scaler = joblib.load(scaler_path)
        numeric = np.load(numeric_path)
        return (
            np.concatenate([np.load(interaction_path), numeric], axis=1),
            fitted_scaler,
        )

    logging.info("Encoding question and answer chunks with MiniLM")
    question_embeddings = encoder.encode(
        frame["text_a"].fillna("").tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    chunks = [answer_chunks(value) for value in frame["text_b"].tolist()]
    flattened = [chunk for sample in chunks for chunk in sample]
    chunk_embeddings = encoder.encode(
        flattened,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    chunk_embeddings = chunk_embeddings.reshape(len(frame), 3, -1)
    answer_mean = chunk_embeddings.mean(axis=1)
    answer_max = chunk_embeddings.max(axis=1)
    interactions = np.concatenate(
        [
            question_embeddings,
            answer_mean,
            answer_max,
            np.abs(question_embeddings - answer_mean),
            question_embeddings * answer_mean,
        ],
        axis=1,
    )
    numeric_raw = engineered_text_features(frame)
    if scaler is None:
        fitted_scaler = StandardScaler()
        numeric = fitted_scaler.fit_transform(numeric_raw)
    else:
        fitted_scaler = scaler
        numeric = fitted_scaler.transform(numeric_raw)
    features = np.concatenate([interactions, numeric], axis=1)
    interaction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(interaction_path, interactions)
    np.save(numeric_path, numeric)
    joblib.dump(fitted_scaler, scaler_path)
    return features, fitted_scaler


def select_improved_c(
    features: np.ndarray,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[float, dict[str, float]]:
    candidates = [0.3, 1.0, 3.0, 10.0]
    splitter = GroupKFold(n_splits=5)
    scores: dict[str, float] = {}
    for c_value in candidates:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
            n_jobs=1,
        )
        values = cross_val_score(
            classifier,
            features,
            y,
            cv=splitter,
            groups=groups,
            scoring="f1_macro",
            n_jobs=1,
        )
        scores[str(c_value)] = float(np.mean(values))
    best = max(scores, key=scores.get)
    return float(best), scores


def evaluate_minilm_model(
    name: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions_path: Path,
    report_path: Path,
    model_path: Path,
    paths: dict[str, Path],
    c_value: float = 3.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, LogisticRegression]:
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(train_features, train["label"])
    predicted = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)
    metrics = metric_dict(test["label"], predicted, LABEL_ORDER)
    metrics.update(
        {
            "model": name,
            "ngram": "n/a",
            "class_weight": "balanced",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
    )
    joblib.dump(classifier, model_path)
    save_classification_report(
        test["label"],
        predicted,
        LABEL_ORDER,
        LABEL_DISPLAY,
        report_path,
    )
    save_predictions(
        test,
        test["label"],
        predicted,
        probabilities,
        classifier.classes_,
        predictions_path,
    )
    save_confusion_plot(
        test["label"],
        predicted,
        LABEL_ORDER,
        LABEL_DISPLAY,
        f"{name}: QEvasion confusion matrix",
        paths["figures"] / f"confusion_{name}.png",
    )
    return metrics, predicted, probabilities, classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QEvasion baseline and improved-model experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "step1.json",
    )
    args = parser.parse_args()
    configure_logging(PROJECT_ROOT / "logs" / "step2_qevasion.log")
    configure_plots()
    config = load_json(args.config)
    paths = paths_for_step("step2")
    metadata = run_metadata()
    started = time.time()

    train = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "qevasion_train.parquet")
    test = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "qevasion_test.parquet")
    train = train.loc[train["label"].isin(LABEL_ORDER)].copy()
    test = test.loc[test["label"].isin(LABEL_ORDER)].copy()
    train = train.drop_duplicates(subset=["text_a", "text_b", "label"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    logging.info("Training rows after normalized deduplication: %d; test rows: %d", len(train), len(test))
    if train["group"].replace("", "").eq("").any():
        train.loc[train["group"].eq(""), "group"] = "unknown"

    save_evasion_eda(train, paths)
    train_text = pair_text(train)
    test_text = pair_text(test)

    comparisons, tfidf_models, _ = run_tfidf_suite(
        train,
        test,
        train_text,
        test_text,
        paths,
    )
    tfidf_frame = pd.DataFrame(comparisons).sort_values("macro_f1", ascending=False)
    best_tfidf_row = tfidf_frame.loc[tfidf_frame["model"].ne("majority_baseline")].iloc[0]
    best_tfidf_name = str(best_tfidf_row["model"])
    best_tfidf_model = tfidf_models[best_tfidf_name]
    grouped_cv = grouped_cross_validation(train, train_text, best_tfidf_name, paths)
    group_clean = evaluate_group_clean_test(
        best_tfidf_model,
        best_tfidf_name,
        train,
        test,
        test_text,
        paths,
    )
    save_confusion_plot(
        test["label"],
        best_tfidf_model.predict(test_text),
        LABEL_ORDER,
        LABEL_DISPLAY,
        f"{best_tfidf_name}: QEvasion confusion matrix",
        paths["figures"] / f"confusion_{best_tfidf_name}.png",
    )

    model_dir = Path(str(config.get("minilm_cache_dir", "")))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not model_dir.exists():
        model_dir = Path(r"D:\Important\3自然语言处理\NLP_Assignment\models_clean")
    encoder = load_sentence_encoder(model_dir)
    baseline_train = encode_baseline_features(
        train,
        encoder,
        paths["features"] / "minilm_baseline_train.npy",
    )
    baseline_test = encode_baseline_features(
        test,
        encoder,
        paths["features"] / "minilm_baseline_test.npy",
    )
    baseline_metrics, baseline_pred, baseline_probs, baseline_model = evaluate_minilm_model(
        "minilm_concat_logreg",
        baseline_train,
        baseline_test,
        train,
        test,
        paths["predictions"] / "predictions_minilm_concat_logreg.csv",
        paths["reports"] / "minilm_concat_logreg.json",
        paths["models"] / "minilm_concat_logreg.joblib",
        paths,
        c_value=3.0,
    )
    comparisons.append(baseline_metrics)

    improved_train, scaler = encode_improved_features(
        train,
        encoder,
        None,
        paths["features"] / "minilm_improved_train_interactions.npy",
        paths["features"] / "minilm_improved_train_numeric.npy",
        paths["features"] / "minilm_improved_scaler.joblib",
    )
    improved_test, _ = encode_improved_features(
        test,
        encoder,
        scaler,
        paths["features"] / "minilm_improved_test_interactions.npy",
        paths["features"] / "minilm_improved_test_numeric.npy",
        paths["features"] / "minilm_improved_scaler.joblib",
    )
    best_c, c_scores = select_improved_c(improved_train, train["label"], train["group"])
    write_json(
        paths["root"] / "minilm_c_selection.json",
        {"selected_c": best_c, "mean_grouped_macro_f1": c_scores},
    )
    improved_metrics, improved_pred, improved_probs, improved_model = evaluate_minilm_model(
        "minilm_chunk_interaction_logreg",
        improved_train,
        improved_test,
        train,
        test,
        paths["predictions"] / "predictions_minilm_chunk_interaction_logreg.csv",
        paths["reports"] / "minilm_chunk_interaction_logreg.json",
        paths["models"] / "minilm_chunk_interaction_logreg.joblib",
        paths,
        c_value=best_c,
    )
    comparisons.append(improved_metrics)
    improved_group_clean = evaluate_group_clean_feature_model(
        improved_model,
        "minilm_chunk_interaction_logreg",
        train,
        test,
        improved_test,
        paths,
    )

    comparison_frame = pd.DataFrame(comparisons).sort_values("macro_f1", ascending=False)
    comparison_frame.to_csv(
        paths["root"] / "table3_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_row = comparison_frame.iloc[0]
    best_name = str(best_row["model"])
    best_predictions_path = paths["predictions"] / f"predictions_{best_name}.csv"
    if best_predictions_path.exists():
        best_predictions = pd.read_csv(best_predictions_path, encoding="utf-8-sig")
    else:
        best_predictions = pd.read_csv(
            paths["predictions"] / "predictions_tfidf_uni_bi_lr_balanced.csv",
            encoding="utf-8-sig",
        )
    best_predictions.to_csv(
        paths["predictions"] / "predictions_best_qevasion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_report = paths["reports"] / f"{best_name}.json"
    if not best_report.exists():
        if best_name == "majority_baseline":
            best_report = paths["reports"] / f"{best_tfidf_name}.json"
        else:
            raise FileNotFoundError(best_report)
    report_data = json.loads(best_report.read_text(encoding="utf-8"))
    class_rows = []
    for label, display in zip(LABEL_ORDER, LABEL_DISPLAY):
        class_info = report_data[display]
        class_rows.append(
            {
                "label": label,
                "precision": class_info["precision"],
                "recall": class_info["recall"],
                "f1": class_info["f1-score"],
                "support": class_info["support"],
            }
        )
    class_rows.append(
        {
            "label": "macro_average",
            "precision": report_data["macro avg"]["precision"],
            "recall": report_data["macro avg"]["recall"],
            "f1": report_data["macro avg"]["f1-score"],
            "support": report_data["macro avg"]["support"],
        }
    )
    pd.DataFrame(class_rows).to_csv(
        paths["root"] / "table4_best_classification_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "dataset": {
            "train_rows_source": 3448,
            "train_rows_after_normalized_deduplication": int(len(train)),
            "normalized_duplicates_removed": int(3448 - len(train)),
            "test_rows": int(len(test)),
        },
        "comparisons": comparison_frame.to_dict(orient="records"),
        "best_model": best_name,
        "grouped_cross_validation": grouped_cv,
        "group_clean_test": group_clean,
        "group_clean_improved_minilm": improved_group_clean,
        "minilm_c_selection": {"selected_c": best_c, "scores": c_scores},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(paths["root"] / "metrics_summary.json", summary)
    logging.info("Best QEvasion model by Macro-F1: %s", best_name)
    log_dataframe(comparison_frame, "QEvasion model comparison")
    finalize_run_metadata(metadata, paths["root"] / "run_info.json")


if __name__ == "__main__":
    main()
