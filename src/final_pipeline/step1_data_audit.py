from __future__ import annotations

import argparse
import logging
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from final_pipeline.common import (
    PROJECT_ROOT,
    configure_logging,
    download_file,
    extract_first_existing,
    load_json,
    normalize_text,
    read_qnli_tsv,
    save_parquet,
    sha256_file,
    write_json,
)


QEVASION_LABEL_BINARY = {
    "Clear Reply": 1,
    "Ambivalent": 0,
    "Clear Non-Reply": 0,
}
QNLI_LABEL_NAME = {
    0: "entailment",
    1: "not_entailment",
}
QNLI_LABEL_BINARY = {
    "entailment": 1,
    "not_entailment": 0,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def adapt_qevasion(qevasion_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    paths = {
        "train": qevasion_dir / "train_clean.csv",
        "test": qevasion_dir / "test_clean.csv",
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing QEvasion files in {qevasion_dir}: {missing}")

    adapted: dict[str, pd.DataFrame] = {}
    source_info: dict[str, object] = {}
    for split, path in paths.items():
        source = read_table(path)
        label_column = "label" if "label" in source.columns else "clarity_label"
        if label_column not in source.columns:
            raise KeyError(f"No clarity label column in {path}")

        frame = pd.DataFrame(
            {
                "dataset": "qevasion",
                "split": split,
                "row_id": range(len(source)),
                "text_a": source["question"].map(normalize_text),
                "text_b": source["interview_answer"].map(normalize_text),
                "label": source[label_column].map(normalize_text),
            }
        )
        frame["label_binary"] = frame["label"].map(QEVASION_LABEL_BINARY).astype("Int64")
        frame["label_name"] = frame["label"]
        frame["group"] = ""
        for source_column, target_column in (
            ("evasion_label", "evasion_label"),
            ("president", "president"),
            ("title", "title"),
            ("date", "date"),
            ("url", "url"),
        ):
            frame[target_column] = (
                source[source_column].map(normalize_text)
                if source_column in source.columns
                else ""
            )
        if "url" in source.columns:
            frame["group"] = source["url"].map(normalize_text)
        if "president" in source.columns:
            frame.loc[frame["group"].eq(""), "group"] = source.loc[
                frame["group"].eq(""), "president"
            ].map(normalize_text)

        adapted[split] = frame
        raw_duplicate_rows = int(
            source.duplicated(
                subset=["question", "interview_answer", label_column],
                keep="first",
            ).sum()
        )
        source_info[split] = {
            "path": str(path),
            "rows": int(len(source)),
            "sha256": sha256_file(path),
            "raw_exact_duplicate_rows": raw_duplicate_rows,
        }

    mapping = {
        "questions": [
            {"source": "question", "target": "text_a"},
            {"source": "interview_answer", "target": "text_b"},
            {"source": "clarity_label", "target": "label"},
        ],
        "binary_mapping": QEVASION_LABEL_BINARY,
    }
    return adapted, {"sources": source_info, "mapping": mapping}


def ensure_qnli_parquet(config: dict[str, object], raw_dir: Path) -> dict[str, Path]:
    train_path = raw_dir / "qnli_train.parquet"
    validation_path = raw_dir / "qnli_validation.parquet"
    qnli_config = config["qnli"]
    if not isinstance(qnli_config, dict):
        raise TypeError("config.qnli must be an object")

    if train_path.exists() and validation_path.exists():
        logging.info("Using cached QNLI parquet files")
        return {"train": train_path, "validation": validation_path, "format": "parquet"}

    try:
        if not train_path.exists():
            download_file(str(qnli_config["primary_train_url"]), train_path)
        if not validation_path.exists():
            download_file(str(qnli_config["primary_validation_url"]), validation_path)
        return {"train": train_path, "validation": validation_path, "format": "parquet"}
    except RuntimeError as exc:
        logging.warning("Primary QNLI download failed: %s", exc)

    archive_path = raw_dir / "QNLI.zip"
    if not archive_path.exists():
        download_file(str(qnli_config["fallback_zip_url"]), archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        train_payload = extract_first_existing(archive, ("/train.tsv", "train.tsv"))
        validation_payload = extract_first_existing(archive, ("/dev.tsv", "dev.tsv"))
    train_frame = read_qnli_tsv(train_payload, "train")
    validation_frame = read_qnli_tsv(validation_payload, "validation")
    save_parquet(train_frame, train_path)
    save_parquet(validation_frame, validation_path)
    return {
        "train": train_path,
        "validation": validation_path,
        "format": "official_zip",
        "archive": archive_path,
    }


def adapt_qnli(raw_files: dict[str, Path]) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    adapted: dict[str, pd.DataFrame] = {}
    source_info: dict[str, object] = {}
    for split in ("train", "validation"):
        path = raw_files[split]
        source = read_table(path)
        expected = {"question", "sentence", "label"}
        if not expected.issubset(source.columns):
            raise KeyError(f"QNLI {split} is missing columns: {sorted(expected - set(source.columns))}")

        numeric_label = pd.to_numeric(source["label"], errors="raise").astype("int64")
        label_name = numeric_label.map(QNLI_LABEL_NAME)
        if label_name.isna().any():
            unknown = sorted(numeric_label[label_name.isna()].unique().tolist())
            raise ValueError(f"Unknown QNLI labels: {unknown}")

        frame = pd.DataFrame(
            {
                "dataset": "qnli",
                "split": split,
                "row_id": source["idx"].astype("int64")
                if "idx" in source.columns
                else range(len(source)),
                "text_a": source["question"].map(normalize_text),
                "text_b": source["sentence"].map(normalize_text),
                "label": numeric_label,
                "label_name": label_name,
                "label_binary": label_name.map(QNLI_LABEL_BINARY).astype("Int64"),
                "group": "qnli",
            }
        )
        adapted[split] = frame
        source_info[split] = {
            "path": str(path),
            "rows": int(len(source)),
            "sha256": sha256_file(path),
        }
        if "archive" in raw_files:
            source_info["archive"] = {
                "path": str(raw_files["archive"]),
                "sha256": sha256_file(raw_files["archive"]),
            }

    mapping = {
        "fields": [
            {"source": "question", "target": "text_a"},
            {"source": "sentence", "target": "text_b"},
            {"source": "label", "target": "label"},
        ],
        "label_names": QNLI_LABEL_NAME,
        "binary_mapping": QNLI_LABEL_BINARY,
    }
    return adapted, {"sources": source_info, "mapping": mapping}


def sample_stratified(
    frame: pd.DataFrame,
    size: int,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    if size >= len(frame):
        result = frame.copy()
        result["sample_split"] = split_name
        return result
    sampled, _ = train_test_split(
        frame,
        train_size=size,
        random_state=seed,
        stratify=frame["label_name"],
    )
    result = sampled.sort_values("row_id").copy()
    result["sample_split"] = split_name
    return result


def length_stats(series: pd.Series) -> dict[str, float | int]:
    values = series.map(normalize_text)
    lengths = values.str.len()
    tokens = values.str.split().str.len()
    return {
        "count": int(len(values)),
        "chars_mean": float(lengths.mean()) if len(values) else 0.0,
        "chars_median": float(lengths.median()) if len(values) else 0.0,
        "chars_p90": float(lengths.quantile(0.90)) if len(values) else 0.0,
        "chars_p95": float(lengths.quantile(0.95)) if len(values) else 0.0,
        "chars_max": int(lengths.max()) if len(values) else 0,
        "tokens_mean": float(tokens.mean()) if len(values) else 0.0,
        "tokens_median": float(tokens.median()) if len(values) else 0.0,
        "tokens_p95": float(tokens.quantile(0.95)) if len(values) else 0.0,
        "tokens_max": int(tokens.max()) if len(values) else 0,
    }


def audit_frame(frame: pd.DataFrame) -> dict[str, object]:
    text_a = frame["text_a"].map(normalize_text)
    text_b = frame["text_b"].map(normalize_text)
    label = frame["label_name"].map(normalize_text)
    missing_mask = text_a.eq("") | text_b.eq("") | label.eq("")
    duplicate_mask = frame.duplicated(subset=["text_a", "text_b", "label"], keep="first")
    counts = label.loc[~label.eq("")].value_counts(dropna=False)
    percentages = (counts / max(int(counts.sum()), 1) * 100).round(4)

    return {
        "rows": int(len(frame)),
        "missing_key_rows": int(missing_mask.sum()),
        "empty_text_a_rows": int(text_a.eq("").sum()),
        "empty_text_b_rows": int(text_b.eq("").sum()),
        "missing_label_rows": int(label.eq("").sum()),
        "exact_duplicate_rows": int(duplicate_mask.sum()),
        "text_a": length_stats(text_a),
        "text_b": length_stats(text_b),
        "label_distribution": {
            str(name): {
                "count": int(count),
                "percent": float(percentages.loc[name]),
            }
            for name, count in counts.items()
        },
        "binary_label_distribution": {
            str(name): int(count)
            for name, count in frame["label_binary"].value_counts(dropna=False).sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and audit QEvasion and QNLI data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "step1.json",
    )
    args = parser.parse_args()

    configure_logging(PROJECT_ROOT / "logs" / "step1_data_audit.log")
    config = load_json(args.config)
    seed = int(config["seed"])
    analysis_train_size = int(config["analysis_train_size"])
    analysis_eval_size = int(config["analysis_eval_size"])

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results" / "step1"
    raw_qnli_dir = PROJECT_ROOT / "data" / "raw" / "qnli"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading and adapting QEvasion data")
    qevasion_dir = Path(str(config["qevasion_dir"]))
    if not qevasion_dir.is_absolute():
        qevasion_dir = PROJECT_ROOT / qevasion_dir
    qevasion_frames, qevasion_info = adapt_qevasion(qevasion_dir)
    for split, frame in qevasion_frames.items():
        save_parquet(frame, processed_dir / f"qevasion_{split}.parquet")

    logging.info("Downloading or loading QNLI")
    qnli_raw = ensure_qnli_parquet(config, raw_qnli_dir)
    qnli_frames, qnli_info = adapt_qnli(qnli_raw)
    for split, frame in qnli_frames.items():
        save_parquet(frame, processed_dir / f"qnli_{split}_full.parquet")

    logging.info("Creating fixed analysis samples")
    qnli_train_sample = sample_stratified(
        qnli_frames["train"],
        analysis_train_size,
        seed,
        "analysis_train",
    )
    qnli_eval_sample = sample_stratified(
        qnli_frames["validation"],
        analysis_eval_size,
        seed,
        "analysis_eval",
    )
    save_parquet(qnli_train_sample, processed_dir / "qnli_analysis_train.parquet")
    save_parquet(qnli_eval_sample, processed_dir / "qnli_analysis_eval.parquet")

    audit_frames = {
        "qevasion_train": qevasion_frames["train"],
        "qevasion_test": qevasion_frames["test"],
        "qnli_analysis_train": qnli_train_sample,
        "qnli_analysis_eval": qnli_eval_sample,
    }
    audits = {name: audit_frame(frame) for name, frame in audit_frames.items()}
    full_audits = {
        "qnli_train_full": audit_frame(qnli_frames["train"]),
        "qnli_validation_full": audit_frame(qnli_frames["validation"]),
    }

    table_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    display_names = {
        "qevasion_train": ("QEvasion", "train"),
        "qevasion_test": ("QEvasion", "test"),
        "qnli_analysis_train": ("QNLI", "analysis-train"),
        "qnli_analysis_eval": ("QNLI", "analysis-eval"),
    }
    for name, audit in audits.items():
        dataset, split = display_names[name]
        table_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "rows": audit["rows"],
                "missing_key_rows": audit["missing_key_rows"],
                "duplicate_rows": audit["exact_duplicate_rows"],
                "avg_text_a_chars": round(audit["text_a"]["chars_mean"], 3),
                "avg_text_b_chars": round(audit["text_b"]["chars_mean"], 3),
                "median_text_a_chars": audit["text_a"]["chars_median"],
                "median_text_b_chars": audit["text_b"]["chars_median"],
                "p95_text_a_chars": audit["text_a"]["chars_p95"],
                "p95_text_b_chars": audit["text_b"]["chars_p95"],
            }
        )
        for label_name, count_info in audit["label_distribution"].items():
            label_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "label": label_name,
                    **count_info,
                }
            )
        for field_name in ("text_a", "text_b"):
            length_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "field": field_name,
                    **audit[field_name],
                }
            )

    pd.DataFrame(table_rows).to_csv(
        results_dir / "table1_data_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(label_rows).to_csv(
        results_dir / "label_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(length_rows).to_csv(
        results_dir / "length_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data_sources = {
        "generated_at_utc": now_utc(),
        "qevasion": qevasion_info,
        "qnli": qnli_info,
        "analysis_sampling": {
            "seed": seed,
            "train_pool_size": int(len(qnli_train_sample)),
            "eval_pool_size": int(len(qnli_eval_sample)),
            "train_stratified_by": "label_name",
            "eval_stratified_by": "label_name",
        },
    }
    write_json(results_dir / "data_sources.json", data_sources)
    write_json(
        results_dir / "field_mapping.json",
        {
            "qevasion": qevasion_info["mapping"],
            "qnli": qnli_info["mapping"],
        },
    )
    write_json(
        results_dir / "data_audit.json",
        {
            "analysis_sets": audits,
            "full_sets": full_audits,
            "data_sources": data_sources,
        },
    )

    logging.info("Step 1 complete")
    logging.info("Statistics: %s", results_dir / "table1_data_statistics.csv")
    logging.info("Full audit: %s", results_dir / "data_audit.json")
    print(pd.DataFrame(table_rows).to_string(index=False))


if __name__ == "__main__":
    main()
