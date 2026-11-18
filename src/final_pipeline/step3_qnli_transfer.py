from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from final_pipeline.common import PROJECT_ROOT, configure_logging, load_json, write_json
from final_pipeline.experiment_utils import (
    QNLI_LABEL_ZH,
    QNLI_LABELS,
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
from final_pipeline.step2_qevasion import load_sentence_encoder


QNLI_DISPLAY = [f"{label} / {QNLI_LABEL_ZH[label]}" for label in QNLI_LABELS]
FRACTIONS = {"10%": 0.10, "30%": 0.30, "100%": 1.00}
SEEDS = [42, 52, 62]


def pair_text(frame: pd.DataFrame) -> pd.Series:
    return frame["text_a"].fillna("") + " [SEP] " + frame["text_b"].fillna("")


def build_tfidf() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
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
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def sample_fraction(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    if fraction >= 1.0:
        result = frame.copy()
        result["_pool_index"] = np.arange(len(result))
        return result
    target_size = max(2, int(round(len(frame) * fraction)))
    sampled, _ = train_test_split(
        frame,
        train_size=target_size,
        random_state=seed,
        stratify=frame["label_name"],
    )
    sampled = sampled.copy()
    sampled["_pool_index"] = sampled.index
    return sampled.sort_values("row_id").reset_index(drop=True)


def transfer_numeric_features(frame: pd.DataFrame) -> np.ndarray:
    question = frame["text_a"].fillna("").astype(str)
    answer = frame["text_b"].fillna("").astype(str)
    q_len = question.str.len().to_numpy()
    a_len = answer.str.len().to_numpy()
    q_words = question.str.split().str.len().to_numpy()
    a_words = answer.str.split().str.len().to_numpy()
    q_marks = question.str.count(r"\?").to_numpy()
    sentence_marks = answer.str.count(r"[.!?]+").to_numpy()
    negation = answer.str.lower().str.count(r"\b(?:no|not|never|none|nobody|nothing)\b").to_numpy()
    q_tokens = question.str.lower().str.findall(r"[a-z0-9]+")
    a_tokens = answer.str.lower().str.findall(r"[a-z0-9]+")
    overlap = [
        len(set(q) & set(a)) / max(len(set(q)), 1)
        for q, a in zip(q_tokens, a_tokens)
    ]
    return np.column_stack(
        [
            np.log1p(q_len),
            np.log1p(a_len),
            np.log1p(q_words),
            np.log1p(a_words),
            np.log1p(np.maximum(a_len, 1) / np.maximum(q_len, 1)),
            np.log1p(q_marks),
            np.log1p(sentence_marks),
            np.log1p(negation),
            overlap,
        ]
    ).astype(np.float32)


def source_pair_interactions(qevasion_frame: pd.DataFrame) -> np.ndarray:
    path = PROJECT_ROOT / "results" / "step2" / "features" / "minilm_improved_train_interactions.npy"
    if not path.exists():
        raise FileNotFoundError("Run step 2 before step 3 to generate MiniLM source features")
    interactions = np.load(path)
    question = interactions[:, :384]
    answer_mean = interactions[:, 384:768]
    cosine = np.sum(question * answer_mean, axis=1, keepdims=True)
    return np.concatenate(
        [
            question,
            answer_mean,
            np.abs(question - answer_mean),
            question * answer_mean,
            cosine,
        ],
        axis=1,
    )


def encode_qnli_pair_features(
    frame: pd.DataFrame,
    encoder,
    interaction_path: Path,
) -> np.ndarray:
    if interaction_path.exists():
        logging.info("Using cached QNLI MiniLM interactions: %s", interaction_path)
        return np.load(interaction_path)
    logging.info("Encoding QNLI questions and sentences")
    question = encoder.encode(
        frame["text_a"].fillna("").tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    sentence = encoder.encode(
        frame["text_b"].fillna("").tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    cosine = np.sum(question * sentence, axis=1, keepdims=True)
    interactions = np.concatenate(
        [
            question,
            sentence,
            np.abs(question - sentence),
            question * sentence,
            cosine,
        ],
        axis=1,
    )
    interaction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(interaction_path, interactions)
    return interactions


def fit_binary_tfidf(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    model = build_tfidf()
    model.fit(pair_text(train_frame), train_frame["label_name"])
    prediction = model.predict(pair_text(eval_frame))
    probabilities = model.predict_proba(pair_text(eval_frame))
    return prediction, probabilities, model


def fit_binary_minilm(
    train_features: np.ndarray,
    train_labels: pd.Series,
    eval_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=4000,
        random_state=42,
    )
    classifier.fit(train_features, train_labels)
    prediction = classifier.predict(eval_features)
    probabilities = classifier.predict_proba(eval_features)
    return prediction, probabilities, classifier


def scaled_minilm_train_eval(
    train_interactions: np.ndarray,
    train_frame: pd.DataFrame,
    eval_interactions: np.ndarray,
    eval_frame: pd.DataFrame,
    source_interactions: np.ndarray | None = None,
    source_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_train = transfer_numeric_features(train_frame)
    numeric_eval = transfer_numeric_features(eval_frame)
    if source_interactions is not None and source_frame is not None:
        source_numeric = transfer_numeric_features(source_frame)
        combined_numeric = np.vstack([source_numeric, numeric_train])
        combined_interactions = np.vstack([source_interactions, train_interactions])
    else:
        combined_numeric = numeric_train
        combined_interactions = train_interactions
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(combined_numeric)
    eval_numeric = scaler.transform(numeric_eval)
    return (
        np.concatenate([combined_interactions, numeric_scaled], axis=1),
        np.concatenate([eval_interactions, eval_numeric], axis=1),
    )


def record_result(
    *,
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    representation: str,
    setting: str,
    fraction: str,
    seed: int,
    train_size: int,
    eval_frame: pd.DataFrame,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    paths: dict[str, Path],
    save_prediction_file: bool,
) -> None:
    metrics = metric_dict(eval_frame["label_name"], prediction, QNLI_LABELS)
    metrics.update(
        {
            "representation": representation,
            "setting": setting,
            "fraction": fraction,
            "seed": seed,
            "train_rows": int(train_size),
            "test_rows": int(len(eval_frame)),
        }
    )
    rows.append(metrics)
    if save_prediction_file:
        safe_name = (
            f"predictions_{representation}_{setting}_{fraction.replace('%', 'pct')}_seed{seed}.csv"
        )
        save_predictions(
            eval_frame,
            eval_frame["label_name"],
            prediction,
            probabilities,
            classes,
            paths["predictions"] / safe_name,
        )


def summarize_efficiency(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["representation", "setting", "fraction"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            train_rows_mean=("train_rows", "mean"),
            runs=("seed", "count"),
        )
        .fillna(0.0)
    )
    order = {"10%": 0, "30%": 1, "100%": 2}
    grouped["fraction_order"] = grouped["fraction"].map(order)
    return grouped.sort_values(
        ["representation", "setting", "fraction_order"]
    ).drop(columns=["fraction_order"])


def plot_length_shift(
    qevasion: pd.DataFrame,
    qnli_train: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": "QEvasion answer",
                    "characters": qevasion["text_b"].fillna("").str.len(),
                }
            ),
            pd.DataFrame(
                {
                    "dataset": "QNLI sentence",
                    "characters": qnli_train["text_b"].fillna("").str.len(),
                }
            ),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(8.2, 4.8))
    sns.boxplot(data=frame, x="dataset", y="characters", showfliers=False)
    plt.title("Length shift between source answers and target sentences")
    plt.xlabel("text_b type")
    plt.ylabel("characters")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "qevasion_qnli_length_shift.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_learning_curves(summary: pd.DataFrame, paths: dict[str, Path]) -> None:
    plot_data = summary.loc[summary["setting"].ne("source_only_direct")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    representations = ["tfidf", "minilm"]
    for axis, representation in zip(axes, representations):
        subset = plot_data.loc[plot_data["representation"].eq(representation)]
        for setting in ("target_only", "source_plus_target"):
            line = subset.loc[subset["setting"].eq(setting)].sort_values("train_rows_mean")
            axis.errorbar(
                line["train_rows_mean"],
                line["macro_f1_mean"],
                yerr=line["macro_f1_std"],
                marker="o",
                capsize=4,
                label=setting,
            )
        axis.set_title(f"{representation}: QNLI data efficiency")
        axis.set_xlabel("training rows")
        axis.set_ylabel("Macro-F1")
        axis.legend()
    plt.tight_layout()
    plt.savefig(paths["figures"] / "qnli_data_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QNLI transfer and data-efficiency experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "step1.json",
    )
    args = parser.parse_args()
    configure_logging(PROJECT_ROOT / "logs" / "step3_qnli_transfer.log")
    configure_plots()
    config = load_json(args.config)
    paths = paths_for_step("step3")
    metadata = run_metadata()
    started = time.time()

    qevasion = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "qevasion_train.parquet")
    qevasion = qevasion.drop_duplicates(subset=["text_a", "text_b", "label"]).reset_index(drop=True)
    qevasion = qevasion.loc[qevasion["label_binary"].notna()].copy()
    qevasion["label_name"] = qevasion["label_binary"].map({1: "entailment", 0: "not_entailment"})

    qnli_train = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "qnli_analysis_train.parquet")
    qnli_eval = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "qnli_analysis_eval.parquet")
    qnli_train = qnli_train.reset_index(drop=True)
    qnli_eval = qnli_eval.reset_index(drop=True)
    logging.info("QEvasion binary source rows: %d", len(qevasion))
    logging.info("QNLI training pool: %d; evaluation: %d", len(qnli_train), len(qnli_eval))
    plot_length_shift(qevasion, qnli_train, paths)

    result_rows: list[dict[str, Any]] = []
    representative_predictions: dict[tuple[str, str, str], dict[str, Any]] = {}

    logging.info("Running TF-IDF direct transfer")
    direct_pred, direct_probs, direct_model = fit_binary_tfidf(qevasion, qnli_eval)
    classes = direct_model.classes_
    record_result(
        rows=result_rows,
        prediction_rows=[],
        representation="tfidf",
        setting="source_only_direct",
        fraction="0%",
        seed=42,
        train_size=len(qevasion),
        eval_frame=qnli_eval,
        prediction=direct_pred,
        probabilities=direct_probs,
        classes=classes,
        paths=paths,
        save_prediction_file=True,
    )
    representative_predictions[("tfidf", "source_only_direct", "0%")] = {
        "prediction": direct_pred,
        "probabilities": direct_probs,
        "classes": classes,
    }
    joblib.dump(direct_model, paths["models"] / "tfidf_source_only_direct.joblib")

    logging.info("Running TF-IDF data-efficiency matrix")
    fraction_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for fraction_name, fraction in FRACTIONS.items():
        for seed in SEEDS:
            key = (fraction_name, seed)
            if key not in fraction_cache:
                fraction_cache[key] = sample_fraction(qnli_train, fraction, seed)
            sampled = fraction_cache[key]
            for setting in ("target_only", "source_plus_target"):
                if setting == "target_only":
                    train_frame = sampled
                else:
                    train_frame = pd.concat([qevasion, sampled], ignore_index=True)
                prediction, probabilities, model = fit_binary_tfidf(train_frame, qnli_eval)
                save_file = seed == 42 or fraction_name == "100%"
                record_result(
                    rows=result_rows,
                    prediction_rows=[],
                    representation="tfidf",
                    setting=setting,
                    fraction=fraction_name,
                    seed=seed,
                    train_size=len(train_frame),
                    eval_frame=qnli_eval,
                    prediction=prediction,
                    probabilities=probabilities,
                    classes=model.classes_,
                    paths=paths,
                    save_prediction_file=save_file,
                )
                if seed == 42:
                    representative_predictions[("tfidf", setting, fraction_name)] = {
                        "prediction": prediction,
                        "probabilities": probabilities,
                        "classes": model.classes_,
                    }
                    joblib.dump(
                        model,
                        paths["models"] / f"tfidf_{setting}_{fraction_name.replace('%', 'pct')}.joblib",
                    )

    model_dir = Path(str(config.get("minilm_cache_dir", "")))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not model_dir.exists():
        model_dir = Path(r"D:\Important\3自然语言处理\NLP_Assignment\models_clean")
    encoder = load_sentence_encoder(model_dir)

    qevasion_interactions = source_pair_interactions(qevasion)
    qnli_train_interactions = encode_qnli_pair_features(
        qnli_train,
        encoder,
        paths["features"] / "qnli_train_interactions.npy",
    )
    qnli_eval_interactions = encode_qnli_pair_features(
        qnli_eval,
        encoder,
        paths["features"] / "qnli_eval_interactions.npy",
    )

    source_numeric = transfer_numeric_features(qevasion)
    source_scaler = StandardScaler().fit(source_numeric)
    source_features = np.concatenate(
        [qevasion_interactions, source_scaler.transform(source_numeric)],
        axis=1,
    )
    eval_numeric_source = source_scaler.transform(transfer_numeric_features(qnli_eval))
    eval_features_source = np.concatenate(
        [qnli_eval_interactions, eval_numeric_source],
        axis=1,
    )

    logging.info("Running MiniLM direct transfer")
    minilm_direct_pred, minilm_direct_probs, minilm_direct_model = fit_binary_minilm(
        source_features,
        qevasion["label_name"],
        eval_features_source,
    )
    record_result(
        rows=result_rows,
        prediction_rows=[],
        representation="minilm",
        setting="source_only_direct",
        fraction="0%",
        seed=42,
        train_size=len(qevasion),
        eval_frame=qnli_eval,
        prediction=minilm_direct_pred,
        probabilities=minilm_direct_probs,
        classes=minilm_direct_model.classes_,
        paths=paths,
        save_prediction_file=True,
    )
    representative_predictions[("minilm", "source_only_direct", "0%")] = {
        "prediction": minilm_direct_pred,
        "probabilities": minilm_direct_probs,
        "classes": minilm_direct_model.classes_,
    }

    logging.info("Running MiniLM data-efficiency matrix")
    for fraction_name, fraction in FRACTIONS.items():
        for seed in SEEDS:
            sampled = fraction_cache[(fraction_name, seed)]
            rows = sampled["_pool_index"].to_numpy()
            sampled_interactions = qnli_train_interactions[rows]
            for setting in ("target_only", "source_plus_target"):
                if fraction_name == "100%" and seed != 42:
                    identical = next(
                        row
                        for row in reversed(result_rows)
                        if row["representation"] == "minilm"
                        and row["setting"] == setting
                        and row["fraction"] == "100%"
                        and row["seed"] == 42
                    )
                    copied = dict(identical)
                    copied["seed"] = seed
                    result_rows.append(copied)
                    continue
                if setting == "target_only":
                    train_features, test_features = scaled_minilm_train_eval(
                        sampled_interactions,
                        sampled,
                        qnli_eval_interactions,
                        qnli_eval,
                    )
                    train_labels = sampled["label_name"]
                    train_size = len(sampled)
                else:
                    train_features, test_features = scaled_minilm_train_eval(
                        sampled_interactions,
                        sampled,
                        qnli_eval_interactions,
                        qnli_eval,
                        source_interactions=qevasion_interactions,
                        source_frame=qevasion,
                    )
                    train_labels = pd.concat(
                        [qevasion["label_name"], sampled["label_name"]],
                        ignore_index=True,
                    )
                    train_size = len(qevasion) + len(sampled)
                prediction, probabilities, model = fit_binary_minilm(
                    train_features,
                    train_labels,
                    test_features,
                )
                save_file = seed == 42
                record_result(
                    rows=result_rows,
                    prediction_rows=[],
                    representation="minilm",
                    setting=setting,
                    fraction=fraction_name,
                    seed=seed,
                    train_size=train_size,
                    eval_frame=qnli_eval,
                    prediction=prediction,
                    probabilities=probabilities,
                    classes=model.classes_,
                    paths=paths,
                    save_prediction_file=save_file,
                )
                if seed == 42:
                    representative_predictions[("minilm", setting, fraction_name)] = {
                        "prediction": prediction,
                        "probabilities": probabilities,
                        "classes": model.classes_,
                    }
                    joblib.dump(
                        model,
                        paths["models"] / f"minilm_{setting}_{fraction_name.replace('%', 'pct')}.joblib",
                    )

    results = pd.DataFrame(result_rows)
    results.to_csv(paths["root"] / "data_efficiency_runs.csv", index=False, encoding="utf-8-sig")
    summary = summarize_efficiency(results)
    summary.to_csv(
        paths["root"] / "data_efficiency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_learning_curves(summary, paths)

    table7_rows = []
    step2_summary = pd.read_csv(
        PROJECT_ROOT / "results" / "step2" / "table3_model_comparison.csv",
        encoding="utf-8-sig",
    )
    best_source = step2_summary.iloc[0]
    table7_rows.append(
        {
            "setting": "original_qevasion_best",
            "train_rows": int(best_source["train_rows"]),
            "test_rows": int(best_source["test_rows"]),
            "accuracy": float(best_source["accuracy"]),
            "macro_f1": float(best_source["macro_f1"]),
            "main_conclusion": "QEvasion 三分类主实验结果",
        }
    )
    for setting, label in (
        ("source_only_direct", "direct_transfer"),
        ("target_only", "target_baseline"),
        ("source_plus_target", "adapted_training"),
    ):
        row = results.loc[
            results["representation"].eq("tfidf")
            & results["setting"].eq(setting)
            & results["fraction"].eq("100%" if setting != "source_only_direct" else "0%")
            & results["seed"].eq(42)
        ].iloc[0]
        table7_rows.append(
            {
                "setting": label,
                "train_rows": int(row["train_rows"]),
                "test_rows": int(row["test_rows"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "main_conclusion": "",
            }
        )
    table7 = pd.DataFrame(table7_rows)
    conclusions = {
        "original_qevasion_best": "源任务主结果，用于参照",
        "direct_transfer": "跨领域直接迁移，衡量零目标标注时的可用性",
        "target_baseline": "仅使用目标域 100% 固定训练池",
        "adapted_training": "源任务全部数据加目标域 100% 训练池",
    }
    table7["main_conclusion"] = table7["setting"].map(conclusions)
    table7.to_csv(
        paths["root"] / "table7_transfer_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for representation in ("tfidf", "minilm"):
        key = (representation, "source_plus_target", "100%")
        stored = representative_predictions[key]
        save_classification_report(
            qnli_eval["label_name"],
            stored["prediction"],
            QNLI_LABELS,
            QNLI_DISPLAY,
            paths["reports"] / f"{representation}_source_plus_target_100pct.json",
        )
        save_confusion_plot(
            qnli_eval["label_name"],
            stored["prediction"],
            QNLI_LABELS,
            QNLI_DISPLAY,
            f"{representation}: QNLI adapted training",
            paths["figures"] / f"confusion_{representation}_adapted_100pct.png",
        )

    output = {
        "data": {
            "qevasion_binary_rows": int(len(qevasion)),
            "qnli_train_pool": int(len(qnli_train)),
            "qnli_eval_rows": int(len(qnli_eval)),
        },
        "fractions": FRACTIONS,
        "seeds": SEEDS,
        "runs": int(len(results)),
        "table7": table7.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(paths["root"] / "metrics_summary.json", output)
    log_dataframe(table7, "Transfer comparison")
    log_dataframe(summary, "Data-efficiency summary")
    finalize_run_metadata(metadata, paths["root"] / "run_info.json")


if __name__ == "__main__":
    main()
