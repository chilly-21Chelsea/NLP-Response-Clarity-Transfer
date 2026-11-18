# Step 1 data audit summary

## Data sources

- QEvasion local cleaned files: 3,448 training rows and 308 official test rows.
- QNLI from `nyu-mll/glue`: 104,743 training rows and 5,463 validation rows.
- Fixed experiment samples: 30,000 training rows and 2,000 evaluation rows,
  stratified by label with seed 42.

The QNLI mirror download succeeded. The fallback is the official GLUE QNLI
archive. Source URLs, SHA-256 hashes, and sampling settings are recorded in
`data_sources.json`.

## Key checks

- All four reporting splits have zero missing rows in `text_a`, `text_b`, or
  the label.
- QNLI fixed samples contain 50.01%/49.99% and 49.45%/50.55% label splits for
  train/evaluation respectively.
- QEvasion raw exact duplicates: 46 training rows. After whitespace
  normalization, 60 training rows duplicate another full text pair.
- QEvasion test has no non-empty `evasion_label`. This supports choosing the
  data-efficiency extension instead of the label-hierarchy extension.
- QEvasion answers average about 1,663 characters, while QNLI sentences
  average about 165 characters in the fixed training pool. Length and domain
  shift are therefore first-order migration risks.

## Reproducibility

`verify_step1.ps1` reruns the data pipeline and compares SHA-256 hashes for
seven deterministic artifacts. The latest result is `pass` with zero changed
files; see `reproducibility_check.json`.

## Primary artifacts

- `table1_data_statistics.csv`: report-ready four-row summary.
- `label_distribution.csv`: class counts and percentages.
- `length_distribution.csv`: character and token length statistics.
- `data_audit.json`: complete audit, including full QNLI splits.
- `field_mapping.json`: QEvasion and QNLI field/label adapters.
