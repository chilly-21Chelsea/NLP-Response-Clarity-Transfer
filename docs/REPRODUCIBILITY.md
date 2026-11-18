# Reproducibility record

## Data controls

- QNLI analysis train pool: 30,000 rows, stratified, seed 42.
- QNLI fixed evaluation set: 2,000 rows, stratified, seed 42.
- QEvasion normalized training rows after duplicate removal: 3,388.
- QNLI data source hashes are stored in `results/step1/data_sources.json`.

## Deterministic verification

`verify_step1.ps1` reruns the first data pipeline and compares SHA-256 hashes for
seven processed-data and statistics files. The final check passed with zero
changes and is stored in `results/step1/reproducibility_check.json`.

## Runtime

- Python 3.13.15.
- Windows 10.
- scikit-learn 1.9.0.
- sentence-transformers 6.1.0.
- PyTorch 2.14.1 CPU.
- Step 2 elapsed time: 253.7 seconds.
- Step 3 elapsed time: 264.8 seconds.

## Random seeds and sampling

- QEvasion split seed: 42.
- QNLI fixed pool and evaluation seed: 42.
- QNLI data-efficiency sampling seeds: 42, 52, 62.
- The 100% setting uses the deterministic full 30,000-row pool, so repeated runs
  have zero sampling variance.

## Report verification

The final DOCX files were exported with Microsoft Word and rendered through
Poppler. Every page was inspected for clipping, table overflow, missing glyphs,
orphaned captions, and page-number placement.
