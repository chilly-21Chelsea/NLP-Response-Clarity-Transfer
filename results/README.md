# Results

## QEvasion

- Best Macro-F1: `minilm_chunk_interaction_logreg`, 44.99%.
- Best Accuracy: `tfidf_uni_bi_lr_none`, 68.51%, but Macro-F1 is 39.08%.
- Grouped CV Macro-F1 for the improved MiniLM model: 49.37%.
- Group-clean test Macro-F1 for the improved MiniLM model: 48.33%.

Primary files:

- `step2/table3_model_comparison.csv`
- `step2/table4_best_classification_report.csv`
- `step2/metrics_summary.json`
- `step2/predictions/predictions_best_qevasion.csv`
- `step2/figures/`

## QNLI transfer

- TF-IDF direct transfer: 36.72% Macro-F1.
- TF-IDF target-only, 100% pool: 58.69%.
- MiniLM direct transfer: 66.20%.
- MiniLM target-only, 10% pool: 76.22%.
- MiniLM target-only, 100% pool: 78.60%.

Primary files:

- `step3/table7_transfer_comparison.csv`
- `step3/data_efficiency_runs.csv`
- `step3/data_efficiency_summary.csv`
- `step3/metrics_summary.json`
- `step3/figures/`

## Error analysis

`step4/table5_error_cases.csv` contains five verified model errors selected from
the best QEvasion predictions. `results/final_metrics.json` combines the final
numbers used by both reports.
