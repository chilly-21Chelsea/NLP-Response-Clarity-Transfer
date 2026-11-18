# Experiment report template contract

## Reference

- DOCX: `D:/codex/2026-11-19/bang/work/source_docx/实验报告：对话问答回避检测与跨场景迁移分析(1).docx`
- SHA-256: `F677A29B8629465247FB283806C69AA02D217D3CB12422508C98CE6863D7B401`
- Page count: 7
- Section count: 1
- Reference render: `D:/codex/2026-11-19/bang/work/source_png/big_requirement-*.png`
- Reference PDF: `D:/codex/2026-11-19/bang/work/source_pdf/实验报告：对话问答回避检测与跨场景迁移分析(1).pdf`
- Package evidence: `document_work/template_evidence`

## Page system

- A4 portrait: 8.27 x 11.69 inches.
- Margins: left 1.25 in, right 1.25 in, top 1.00 in, bottom 1.00 in.
- One section, no odd/even header variation.
- Cover page is page 1. The remaining form pages use bordered single-column tables.

## Typography

- Normal body style: 10.5 pt, justified.
- Cover title: 宋体 22 pt bold, centered, character-spaced.
- Cover metadata: 14 pt bold.
- Cover footer text: 12 pt bold, centered.
- The source uses direct formatting heavily; preserve visible hierarchy rather than relying on named heading styles.

## Components

- Cover with 深圳大学实验报告 title and metadata lines for course, project, school, major, instructor, reporter, student ID, class, experimental period, and submission date.
- Five-row requirement and task-definition table.
- Three-row conclusion, instructor review, and notes table.
- Required evidence areas: tables 1 to 7, three or more captioned figures with explanations, five error cases, and a run screenshot.

## Content flow

1. Cover.
2. Experiment purpose and task definition.
3. Data preparation, EDA, model baselines, training configuration.
4. Results, class-level metrics, confusion matrices, error analysis.
5. Cross-scenario transfer and data-efficiency extension.
6. Conclusion, reproduction instructions, and submission file list.

## Editable slots

- Cover metadata: fill course, project, instructor, reporter, student ID, and dates; leave unknown class value blank.
- All requirement tables: replace placeholder content with evidence-backed report content.
- Figure slots: insert verified project figures and captions of at least 50 Chinese characters each.
- Error-analysis table: five actual model errors.
- Conclusion: at least 300 Chinese characters.
- Instructor review and grading area: preserve empty.

## Preserve-only content

- A4 page geometry and margins.
- Cover identity and institutional wording.
- 教务部制 and report notes.
- Instructor review and grading area.
- No headers or footers beyond source behavior.

## Fidelity gates

- Source remains byte-for-byte unchanged.
- Cover remains one page.
- Tables use visible light gray borders and repeating headers when split.
- No text clipping, overlap, missing glyphs, or isolated captions.
- Every required table and figure is present.
- Final render is checked page by page through Word PDF export and PNG rasterization.
