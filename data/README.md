# Data

Full source data and processed tensors are intentionally excluded from Git.

## QEvasion

Run `run_download_qevasion.ps1` from the repository root. The script downloads
`ailsntua/QEvasion` from the configured Hugging Face endpoint and writes:

```text
data/qevasion/train_clean.csv
data/qevasion/test_clean.csv
data/qevasion/download_info.json
```

## QNLI

`step1_data_audit.py` downloads `nyu-mll/glue` configuration `qnli` from the
Hugging Face mirror. If that fails, it downloads the official GLUE QNLI
archive. Source URLs, SHA-256 hashes, row counts, and fixed-sample settings are
written to `results/step1/data_sources.json`.

The fixed analysis samples are:

- 30,000 stratified training rows.
- 2,000 stratified evaluation rows.
- Sampling seed 42.
