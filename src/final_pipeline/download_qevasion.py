from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from final_pipeline.common import PROJECT_ROOT, configure_logging, write_json


def main() -> None:
    configure_logging(PROJECT_ROOT / "logs" / "download_qevasion.log")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    output_dir = PROJECT_ROOT / "data" / "qevasion"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "ailsntua/QEvasion",
        cache_dir=str(PROJECT_ROOT / "data" / "hf_cache"),
    )

    metadata: dict[str, object] = {"dataset": "ailsntua/QEvasion", "splits": {}}
    target_names = {"train": "train_clean.csv", "test": "test_clean.csv"}
    for split_name, filename in target_names.items():
        frame = dataset[split_name].to_pandas()
        required = ["question", "interview_answer", "clarity_label"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"QEvasion {split_name} is missing columns: {missing}")
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
        metadata["splits"][split_name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "path": str(output_dir / filename),
        }
        logging.info("Saved %s (%d rows)", output_dir / filename, len(frame))
    write_json(output_dir / "download_info.json", metadata)


if __name__ == "__main__":
    main()
