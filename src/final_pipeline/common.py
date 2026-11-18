from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    attempts: int = 4,
    timeout_seconds: int = 90,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 codex-nlp-pipeline/1.0"}

    for attempt in range(1, attempts + 1):
        try:
            logging.info("Downloading %s (attempt %d/%d)", url, attempt, attempts)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty")
            temporary.replace(destination)
            logging.info("Saved %s (%d bytes)", destination, destination.stat().st_size)
            return
        except (OSError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"Unable to download {url}: {exc}") from exc
            wait_seconds = 2**attempt
            logging.warning("Download failed: %s; retrying in %ds", exc, wait_seconds)
            time.sleep(wait_seconds)


def extract_first_existing(archive: zipfile.ZipFile, suffixes: tuple[str, ...]) -> bytes:
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/").lower()
        if any(normalized.endswith(suffix.lower()) for suffix in suffixes):
            return archive.read(info)
    raise FileNotFoundError(f"No matching file in archive: {suffixes}")


def read_qnli_tsv(payload: bytes, split: str) -> pd.DataFrame:
    from io import BytesIO

    frame = pd.read_csv(BytesIO(payload), sep="\t", header=None, dtype=str)
    if frame.shape[1] < 3:
        raise ValueError(f"Unexpected QNLI {split} layout: {frame.shape}")

    # The official GLUE TSV may include a header or an index column.
    first_row = [str(value).strip().lower() for value in frame.iloc[0].tolist()]
    if "question" in first_row or "sentence" in first_row:
        frame = frame.iloc[1:].reset_index(drop=True)

    if frame.shape[1] >= 4 and frame.iloc[:, 3].astype(str).str.fullmatch(r"\d+").all():
        frame = frame.iloc[:, :3]

    frame = frame.iloc[:, :3].copy()
    frame.columns = ["question", "sentence", "label"]
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int64")
    frame["idx"] = range(len(frame))
    return frame


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
