from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

from final_pipeline.common import PROJECT_ROOT, configure_logging, write_json


OUTPUTS = Path(os.environ.get("NLP_OUTPUTS", str(PROJECT_ROOT / "outputs")))
DOC_WORK = PROJECT_ROOT / "document_work"
TEMPLATES = PROJECT_ROOT / "templates"
ASSETS = DOC_WORK / "assets"
REPORTS = OUTPUTS / "reports"
RESULT_STEP1 = PROJECT_ROOT / "results" / "step1"
RESULT_STEP2 = PROJECT_ROOT / "results" / "step2"
RESULT_STEP3 = PROJECT_ROOT / "results" / "step3"
RESULT_STEP4 = PROJECT_ROOT / "results" / "step4"

BLACK = RGBColor(0, 0, 0)
DARK_BLUE = "1F4E79"
LIGHT_BLUE = "EAF2F8"
LIGHT_GRAY = "D9D9D9"
WHITE = RGBColor(255, 255, 255)

STUDENT_NAME = "林楷轩"
STUDENT_ID = "2025152024"


def set_run_font(run, family: str, size: float, bold: bool | None = None) -> None:
    run.font.name = family
    run._element.rPr.rFonts.set(qn("w:eastAsia"), family)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def set_paragraph_format(
    paragraph,
    *,
    before: float = 0,
    after: float = 6,
    line: float = 1.45,
    first_indent: bool = True,
) -> None:
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    paragraph.paragraph_format.line_spacing = line
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(21)


def style_document(document: Document, *, body_size: float = 10.5) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(body_size)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(6)

    for style_name, size, before, after in (
        ("Heading 1", 16, 14, 8),
        ("Heading 2", 13.5, 10, 5),
        ("Heading 3", 11.5, 8, 4),
    ):
        try:
            style = document.styles[style_name]
        except KeyError:
            style = document.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = "黑体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.first_line_indent = Pt(0)
    try:
        title_style = document.styles["Title"]
    except KeyError:
        title_style = document.styles.add_style("Title", WD_STYLE_TYPE.PARAGRAPH)
    title_style.font.name = "黑体"
    title_style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title_style.font.size = Pt(18)
    title_style.font.bold = True
    title_style.font.color.rgb = BLACK


def move_body_after(document: Document, anchor, *, remove_anchor: bool = False) -> None:
    body = document._element.body
    remove = False
    for child in list(body):
        if child is anchor:
            remove = True
            if remove_anchor:
                body.remove(child)
            continue
        if remove:
            if child.tag != qn("w:sectPr"):
                body.remove(child)


def add_body_paragraph(
    document: Document,
    text: str,
    *,
    bold_lead: str | None = None,
    size: float = 10.5,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = align
    set_paragraph_format(paragraph, line=1.45, after=6, first_indent=align == WD_ALIGN_PARAGRAPH.JUSTIFY)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, "宋体", size, True)
        rest = paragraph.add_run(text[len(bold_lead) :])
        set_run_font(rest, "宋体", size, False)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, "宋体", size, False)


def add_heading(document: Document, text: str, level: int = 1, numbered: bool = False) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    size = {1: 16, 2: 13.5, 3: 11.5}[level]
    set_run_font(run, "黑体", size, True)
    if numbered:
        run.font.color.rgb = BLACK


def set_cell_fill(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 90, start: int = 100, bottom: int = 90, end: int = 100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "6")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), LIGHT_GRAY)


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    prop = OxmlElement("w:tblHeader")
    prop.set(qn("w:val"), "true")
    tr_pr.append(prop)


def set_no_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    prop = OxmlElement("w:cantSplit")
    prop.set(qn("w:val"), "true")
    tr_pr.append(prop)


def style_cell_text(cell, *, bold: bool = False, color: RGBColor = BLACK, size: float = 9.2, center: bool = False) -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_margins(cell)
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.1
        for run in paragraph.runs:
            set_run_font(run, "宋体", size, bold)
            run.font.color.rgb = color


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.keep_with_next = False
    paragraph.paragraph_format.keep_together = True
    run = paragraph.add_run(text)
    set_run_font(run, "宋体", 9.2, False)


def add_data_table(
    document: Document,
    headers: list[str],
    rows: Iterable[Iterable[object]],
    widths: list[float],
    *,
    font_size: float = 8.8,
    center_columns: set[int] | None = None,
) -> None:
    center_columns = center_columns or set()
    data = list(rows)
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    for index, value in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = Inches(widths[index])
        cell.text = str(value)
        set_cell_fill(cell, DARK_BLUE)
        style_cell_text(cell, bold=True, color=WHITE, size=font_size, center=True)
    set_repeat_header(table.rows[0])
    set_no_split(table.rows[0])
    for row_index, values in enumerate(data):
        row = table.add_row()
        set_no_split(row)
        for col_index, value in enumerate(values):
            cell = row.cells[col_index]
            cell.width = Inches(widths[col_index])
            cell.text = "" if value is None else str(value)
            if row_index % 2 == 1:
                set_cell_fill(cell, LIGHT_BLUE)
            style_cell_text(
                cell,
                bold=False,
                color=BLACK,
                size=font_size,
                center=col_index in center_columns,
            )
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(
    document: Document,
    path: Path,
    caption: str,
    *,
    width: float,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run()
    run.add_picture(str(path), width=Inches(width))
    add_caption(document, caption)


def add_bullets(document: Document, items: list[str], *, size: float = 10.5) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.3
        paragraph.paragraph_format.first_line_indent = Pt(0)
        run = paragraph.add_run(item)
        set_run_font(run, "宋体", size, False)


def add_page_number_footer(document: Document, label: str = "第 ") -> None:
    for section in document.sections:
        footer = section.footer
        paragraph = footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.clear()
        prefix = paragraph.add_run(label)
        set_run_font(prefix, "宋体", 9, False)
        run = paragraph.add_run()
        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " PAGE "
        fld_sep = OxmlElement("w:fldChar")
        fld_sep.set(qn("w:fldCharType"), "separate")
        text = OxmlElement("w:t")
        text.text = "1"
        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        run._r.extend([fld_begin, instr, fld_sep, text, fld_end])
        set_run_font(run, "宋体", 9, False)
        suffix = paragraph.add_run(" 页")
        set_run_font(suffix, "宋体", 9, False)


def create_method_pipeline(path: Path) -> None:
    width, height = 1800, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 31)
    body_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    small_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
    boxes = [
        (50, 125, 320, 320, "QEvasion\n问题 + 回答\n三分类"),
        (410, 125, 690, 320, "统一字段适配\n text_a / text_b\n label / group"),
        (780, 80, 1080, 365, "特征与模型\n词级 TF-IDF\nMiniLM 分块交互"),
        (1170, 80, 1470, 365, "统一评价\nAccuracy\nMacro-F1 / 混淆矩阵"),
        (1530, 125, 1770, 320, "QNLI 迁移\n10% / 30% / 100%"),
    ]
    draw.text((520, 20), "Unified experiment pipeline", font=title_font, fill="#1F2937")
    for index, (x1, y1, x2, y2, label) in enumerate(boxes):
        fill = "#F2F6FA" if index % 2 == 0 else "#FFF7E8"
        draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=fill, outline="#1F4E79", width=4)
        lines = label.split("\n")
        total_height = len(lines) * 34
        y = (y1 + y2 - total_height) // 2
        for line_index, line in enumerate(lines):
            font = body_font if line_index == 0 else small_font
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text(
                (x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2, y),
                line,
                font=font,
                fill="#111827",
            )
            y += 34
    for start, end in (((320, 222), (410, 222)), ((690, 222), (780, 222)), ((1080, 222), (1170, 222)), ((1470, 222), (1530, 222))):
        draw.line([start, end], fill="#1F4E79", width=6)
        draw.polygon([(end[0], end[1]), (end[0] - 18, end[1] - 11), (end[0] - 18, end[1] + 11)], fill="#1F4E79")
    draw.text(
        (50, 430),
        "Source-to-target label mapping: Clear Reply -> contain answer; Ambivalent and Clear Non-Reply -> no clear answer.",
        font=small_font,
        fill="#374151",
    )
    draw.text(
        (50, 475),
        "Evaluation controls: fixed seed, fixed evaluation set, interview-URL group split, no target leakage in reports.",
        font=small_font,
        fill="#374151",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def fill_report_cover(document: Document) -> None:
    replacements = {
        5: "      课程名称：             自然语言处理                      ",
        7: "      实验项目名称：    对话问答回避检测与跨场景迁移分析             ",
        13: "      指导教师：                陈俊扬                     ",
        15: f"      报告人：   {STUDENT_NAME}       学号：{STUDENT_ID}      班级：________",
    }
    for index, text in replacements.items():
        paragraph = document.paragraphs[index]
        paragraph.clear()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        set_run_font(run, "宋体", 14, True)
    for index in range(2, 22):
        paragraph = document.paragraphs[index]
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    for index in (23, 22, 20, 18, 4):
        paragraph = document.paragraphs[index]
        paragraph._element.getparent().remove(paragraph._element)


def add_experiment_body(document: Document, figures: dict[str, Path]) -> None:
    document.add_page_break()
    paragraph = document.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("对话问答回避检测与跨场景迁移分析实验报告")
    set_run_font(run, "黑体", 18, True)
    run.font.color.rgb = BLACK
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(10)
    run = subtitle.add_run("QEvasion 三分类、QNLI 迁移与数据效率实验")
    set_run_font(run, "宋体", 10.5, False)

    add_heading(document, "一、实验目的与任务定义", 1)
    add_body_paragraph(
        document,
        "本实验研究一个目标问题及其对应回答是否真正提供了问题所需的信息。任务输入为 question 与 interview_answer，输出为 Clear Reply、Ambivalent 和 Clear Non-Reply 三类。该任务不同于情感分类，模型需要判断回答与问题之间的信息关系，而不仅是词面正负倾向。",
    )
    add_body_paragraph(
        document,
        "实验目标包括：建立可复现的数据清洗与统一字段流程；完成 QEvasion 的探索性分析；比较多数类、TF-IDF 与预训练表示；使用 Macro-F1、类别级指标和混淆矩阵分析模型；通过 QNLI 检验字段适配和跨场景迁移边界；比较 10%、30% 和 100% 目标域数据的使用效率。",
    )
    add_heading(document, "1.1 标签体系", 2)
    add_data_table(
        document,
        ["高层标签", "中文含义", "判断重点"],
        [
            ("Clear Reply", "明确回答", "回答直接处理问题，通常只有一个清晰解释。"),
            ("Ambivalent", "含糊或矛盾", "回答包含部分信息、绕圈、隐式回答或主题转移。"),
            ("Clear Non-Reply", "明确不回答", "拒绝回答、声称不知道或明确不提供所需信息。"),
        ],
        [1.35, 1.35, 3.05],
        center_columns={0, 1},
    )
    add_body_paragraph(
        document,
        "训练集的细粒度 evasion_label 包含 Explicit、Implicit、Dodging、General、Deflection、Partial/half-answer、Declining to answer、Claims ignorance 和 Clarification。官方测试集的 evasion_label 为空，因此本实验不把细粒度标签作为监督目标，只用于训练集分布分析和错误解释。",
    )

    add_heading(document, "二、数据来源与预处理", 1)
    add_body_paragraph(
        document,
        "QEvasion 来自 Hugging Face 数据集 ailsntua/QEvasion，训练集 3,448 条、官方测试集 308 条。数据中的 question、interview_answer 和 clarity_label 均无缺失。原始训练集有 46 条完全重复行；按统一空白规范化后，模型输入层面的重复行为 60 条。正式训练删除规范化重复，得到 3,388 条。",
    )
    add_body_paragraph(
        document,
        "迁移数据选用 GLUE 的 QNLI。完整训练集 104,743 条、验证集 5,463 条，字段为 question、sentence 和 label。为控制 CPU 训练成本并保证重复实验一致，固定随机种子 42，从训练集分层抽取 30,000 条作为 100% 训练池，从验证集分层抽取 2,000 条作为固定评估集。QNLI 样本没有关键字段缺失或完全重复。",
    )
    table1 = pd.read_csv(RESULT_STEP1 / "table1_data_statistics.csv", encoding="utf-8-sig")
    add_caption(document, "表1  数据统计记录")
    add_data_table(
        document,
        ["数据划分", "样本数", "关键缺失", "平均 text_a 字符", "平均 text_b 字符"],
        [
            (row.dataset + "-" + row.split, f"{int(row.rows):,}", int(row.missing_key_rows), f"{row.avg_text_a_chars:.1f}", f"{row.avg_text_b_chars:.1f}")
            for row in table1.itertuples()
        ],
        [1.45, 0.85, 0.85, 1.2, 1.3],
        center_columns={0, 1, 2, 3, 4},
    )
    add_body_paragraph(
        document,
        "统一字段适配器把 QEvasion 的 question、interview_answer 和 clarity_label 映射为 text_a、text_b 和 label；把 QNLI 的 question、sentence 和二元标签映射到相同字段。QEvasion 的二分类辅助映射为 Clear Reply 对应包含答案，Ambivalent 和 Clear Non-Reply 对应不含明确答案。",
    )
    add_caption(document, "统一字段适配说明")
    add_data_table(
        document,
        ["统一字段", "QEvasion", "QNLI", "用途"],
        [
            ("text_a", "question", "question", "目标问题"),
            ("text_b", "interview_answer", "sentence", "待判断回答或句子"),
            ("label", "clarity_label", "entailment 标签", "监督目标"),
            ("group", "采访 URL", "固定数据来源", "分组与泄漏检查"),
        ],
        [0.85, 1.25, 1.2, 1.65],
        font_size=8.0,
        center_columns={0},
    )

    add_heading(document, "三、探索性数据分析", 1)
    add_body_paragraph(
        document,
        "本节从高层类别分布、细粒度回避策略和文本长度三个角度检查数据。统计结果用于判断类别权重、截断策略和验证切分方式，不把官方测试集的信息用于训练。",
    )
    add_figure(
        document,
        figures["clarity"],
        "图1  QEvasion 高层清晰度标签分布。Ambivalent 为训练集和测试集中最多的类别，Clear Non-Reply 最少，类别不均衡明显。该分布说明 Accuracy 容易被多数类抬高，主要评价指标应采用 Macro-F1，并结合少数类召回率。",
        width=4.8,
    )
    add_figure(
        document,
        figures["evasion"],
        "图2  QEvasion 训练集细粒度回避策略分布。Explicit 数量最多，Dodging、Implicit 和 General 也占较大比例，Declining to answer、Claims ignorance 和 Clarification 数量较少。少数策略样本不足，适合用于解释高层类别内部差异，不适合单独训练复杂分类器。",
        width=5.1,
    )
    add_figure(
        document,
        figures["length"],
        "图3  不同清晰度标签下的回答长度分布。回答整体远长于问题，且长回答常包含背景、转折和补充说明。模型如果只依赖开头短句，可能把含糊回答误判为明确回答；如果只用截断输入，又会遗漏回答尾部的重要信息。",
        width=4.9,
    )
    add_body_paragraph(
        document,
        "EDA 还发现官方训练集与测试集存在同一采访 URL 跨集合现象。为避免来源泄漏，交叉验证采用按 URL 分组的 GroupKFold；官方测试集作为主结果保留，并额外报告删除同源采访后的敏感性结果。",
    )

    add_heading(document, "四、模型与训练配置", 1)
    add_body_paragraph(
        document,
        "多数类基线始终预测训练集中数量最多的 Ambivalent。TF-IDF 实验比较 unigram 与 unigram+bigram，并分别记录不使用类别权重和使用 class_weight='balanced' 的结果。分类器为逻辑回归，C=2.0，最大迭代次数 4,000，随机种子 42。",
    )
    add_body_paragraph(
        document,
        "预训练表示使用 sentence-transformers/all-MiniLM-L6-v2。拼接基线把问题、分隔符和截断到约 2,200 字符的回答整体编码。改进模型分别编码问题以及回答的首段、中段、尾段，每段最多 1,200 字符，再组合问题向量、回答均值、回答最大值、绝对差、乘积和 11 维文本统计特征。分类器为逻辑回归，C 通过按采访 URL 分组的 5 折交叉验证在 0.3、1、3、10 中选择，最终 C=1。",
    )
    document.add_page_break()
    add_caption(document, "表2  训练配置记录")
    add_data_table(
        document,
        ["模型", "表示", "词级范围", "类别权重", "关键参数", "种子"],
        [
            ("多数类基线", "标签频数", "无", "无", "始终预测 Ambivalent", "42"),
            ("TF-IDF uni", "词级 TF-IDF", "1-gram", "无", "LR C=2.0", "42"),
            ("TF-IDF uni balanced", "词级 TF-IDF", "1-gram", "balanced", "LR C=2.0", "42"),
            ("TF-IDF uni+bi", "词级 TF-IDF", "1-2 gram", "无", "LR C=2.0", "42"),
            ("TF-IDF uni+bi balanced", "词级 TF-IDF", "1-2 gram", "balanced", "LR C=2.0", "42"),
            ("MiniLM 拼接", "384 维句向量", "整体输入", "balanced", "LR C=3.0", "42"),
            ("MiniLM 分块交互", "3 个回答片段", "分块池化", "balanced", "LR C=1.0", "42"),
        ],
        [1.25, 1.15, 0.7, 0.8, 1.2, 0.45],
        font_size=8.3,
        center_columns={2, 3, 5},
    )
    add_body_paragraph(
        document,
        "所有模型使用同一训练集和同一个官方测试集。训练环境为 Python 3.13.15、Windows 10、scikit-learn 1.9.0、CPU 运行；主实验耗时 253.7 秒，QNLI 迁移实验耗时 264.8 秒。完整运行信息保存在 results/step2/run_info.json 和 results/step3/run_info.json。",
    )

    add_heading(document, "五、模型结果与类别级分析", 1)
    table3 = pd.read_csv(RESULT_STEP2 / "table3_model_comparison.csv", encoding="utf-8-sig")
    add_caption(document, "表3  QEvasion 模型总体性能（官方 test，n=308）")
    add_data_table(
        document,
        ["模型", "Accuracy", "Macro-P", "Macro-R", "Macro-F1"],
        [
            (
                row.model,
                f"{row.accuracy:.2%}",
                f"{row.macro_precision:.2%}",
                f"{row.macro_recall:.2%}",
                f"{row.macro_f1:.2%}",
            )
            for row in table3.itertuples()
        ],
        [2.25, 0.82, 0.82, 0.82, 0.82],
        font_size=8.3,
        center_columns={1, 2, 3, 4},
    )
    add_body_paragraph(
        document,
        "改进后的 MiniLM 分块交互模型取得最高 Macro-F1 44.99%，Accuracy 51.30%。词级 TF-IDF balanced 模型 Macro-F1 为 40.94%，uni+bigram balanced 为 40.80%，与原拼接 MiniLM 的 35.76% 相比，分块和文本特征带来的提升为 9.24 个百分点。原拼接 MiniLM 低于 TF-IDF 的主要原因是回答过长，直接截断丢失了中段和尾部的语义信息。",
    )
    add_figure(
        document,
        figures["model_comparison"],
        "图4  全部 QEvasion 基线的 Accuracy 与 Macro-F1 对比。多数类基线 Accuracy 为 66.88%，但 Macro-F1 只有 26.72%；不使用类别权重的 uni+bigram 模型 Accuracy 为 68.51%，Macro-F1 仍仅 39.08%。这说明整体准确率主要被 Ambivalent 类别支撑。",
        width=5.25,
    )
    add_caption(document, "表4  最佳模型类别级结果")
    table4 = pd.read_csv(RESULT_STEP2 / "table4_best_classification_report.csv", encoding="utf-8-sig")
    add_data_table(
        document,
        ["类别", "Precision", "Recall", "F1", "支持数"],
        [
            (
                row.label,
                f"{row.precision:.2%}",
                f"{row.recall:.2%}",
                f"{row.f1:.2%}",
                f"{int(row.support)}",
            )
            for row in table4.itertuples()
        ],
        [2.1, 0.85, 0.85, 0.85, 0.75],
        center_columns={1, 2, 3, 4},
    )
    add_figure(
        document,
        figures["confusion"],
        "图5  最佳 MiniLM 分块交互模型的混淆矩阵。对角线为正确预测数，主要错误是 Ambivalent 被判为 Clear Reply，以及 Clear Reply 被判为 Ambivalent。Clear Non-Reply 的召回率提升到 52.17%，但精确率仍仅 24.49%，说明模型能捕捉部分拒答信号，却容易把含糊回答也归入不回答。",
        width=4.85,
    )
    add_body_paragraph(
        document,
        "按采访 URL 分组的 5 折交叉验证中，改进 MiniLM 在训练集内部的 Macro-F1 均值为 49.37%。删除与训练集同源采访后的 190 条测试样本上，Macro-F1 为 48.33%，Accuracy 为 54.21%。敏感性结果没有出现显著下降，说明主结论不是单纯由同源采访泄漏造成，但正式评价仍应保留分组验证。",
    )
    add_body_paragraph(
        document,
        "分组清洁子集比官方 test 少 118 条同源样本，剩余 190 条中三类别的支持数更不均衡。该子集上的提升说明模型没有只依赖相同采访中的重复议题；但由于样本量下降，结果仍有区间波动。正式训练时继续使用官方 train，所有敏感性分析只用于解释来源群体影响。",
    )

    add_heading(document, "六、错误分析与典型案例", 1)
    add_figure(
        document,
        figures["errors"],
        "图6  QEvasion 主要错误类型统计。Ambivalent 被判为 Clear Reply 是最常见错误，共 73 例；其次是 Clear Reply 被判为 Ambivalent，共 29 例。错误集中在回答前半段具有短明确句、后半段继续铺陈，以及拒答、部分回答和背景转移边界。",
        width=5.2,
    )
    table5 = pd.read_csv(RESULT_STEP4 / "table5_error_cases.csv", encoding="utf-8-sig")
    add_caption(document, "表5  典型误判样本")
    add_data_table(
        document,
        ["序号", "问题摘要", "真实标签", "预测标签", "错误类型", "原因"],
        [
            (
                index,
                (
                    f"{row.question_summary_zh}；{row.answer_summary_zh}"
                    if len(f"{row.question_summary_zh}；{row.answer_summary_zh}") <= 92
                    else f"{row.question_summary_zh}；{row.answer_summary_zh[:55]}…"
                ),
                row.true_label,
                row.predicted_label,
                row.error_type,
                row.reason_zh if len(row.reason_zh) <= 62 else row.reason_zh[:59] + "…",
            )
            for index, row in enumerate(table5.itertuples(), start=1)
        ],
        [0.38, 2.35, 0.82, 0.88, 0.9, 1.05],
        font_size=7.2,
        center_columns={0, 2, 3},
    )
    add_body_paragraph(
        document,
        "这些错误表明，标签边界与回答结构共同影响判断。长回答中明确的短句容易被平均池化稀释；拒绝给出完整信息但随后解释制度标准时，模型容易判定为明确不回答；回答转向相邻议题时，模型又可能把与问题相关但未回答问题的内容当成明确回答。",
    )

    add_heading(document, "七、跨场景迁移与数据效率实验", 1)
    add_body_paragraph(
        document,
        "QNLI 的目标问题是判断 sentence 是否包含 question 所需答案。直接迁移时，QEvasion 模型对问题与句子预测三分类，再把 Clear Reply 映射为 entailment，把其他两类映射为 not_entailment。目标域训练和适配训练则先把 QEvasion 标签映射为二分类，保证所有迁移设置使用相同的标签口径。",
    )
    add_body_paragraph(
        document,
        "直接迁移衡量的是零目标标注条件下的可用性，不代表模型已经理解 QNLI 的任务定义。目标域基线衡量同一接口在新数据上的上限；适配训练加入全部源任务样本，用来检验额外源数据是否帮助新场景。三种设置分开报告，避免把源任务、目标任务和混合训练的数字放在同一标签含义下比较。",
    )
    add_body_paragraph(
        document,
        "QNLI 的 sentence 可能包含答案，也可能包含与问题相关但不充分的事实。该标签与 QEvasion 的 Clear Reply 只在“是否提供所需答案”这一点上相近；Ambivalent 和 Clear Non-Reply 仍然共享不含明确答案的负类，但二者在目标任务中没有对应细分。本文因此只使用二分类映射，不把三分类结果直接当作 QNLI 三类输出。",
    )
    add_body_paragraph(
        document,
        "正式运行前先比较了两边的字段分布和标签支持数。QNLI 固定评估集接近 50%/50%，没有多数类可被简单利用的空间；QEvasion 的官方测试集则为 25.65% Clear Reply、66.88% Ambivalent、7.47% Clear Non-Reply。两个任务的标签平衡程度不同，因此迁移表主要使用 Accuracy 与 Macro-F1，并把源任务三分类结果标明为不可直接横向比较。",
    )
    add_figure(
        document,
        figures["length_shift"],
        "图7  QEvasion 回答与 QNLI 句子的长度分布对比。QEvasion 回答平均约 1,663 字符，QNLI 句子平均约 165 字符，长度相差约一个数量级。源任务依赖长回答中的背景、转折和尾部信息，目标任务的输入更短，因此直接迁移的主要风险来自领域词汇、回答结构和标签边界同时变化。",
        width=4.95,
    )
    add_caption(document, "表7  原任务与迁移任务对比")
    table7 = pd.read_csv(RESULT_STEP3 / "table7_transfer_comparison.csv", encoding="utf-8-sig")
    add_data_table(
        document,
        ["实验设置", "训练样本", "测试样本", "Accuracy", "Macro-F1", "主要结论"],
        [
            (
                row.setting,
                f"{int(row.train_rows):,}",
                f"{int(row.test_rows):,}",
                f"{row.accuracy:.2%}",
                f"{row.macro_f1:.2%}",
                row.main_conclusion,
            )
            for row in table7.itertuples()
        ],
        [1.15, 0.65, 0.65, 0.75, 0.75, 1.35],
        font_size=8.0,
        center_columns={1, 2, 3, 4},
    )
    add_body_paragraph(
        document,
        "直接迁移的 Accuracy 只比决策边界附近的随机猜略高，Macro-F1 低于目标域训练二十多个百分点。目标域数据达到 30,000 条后，词级模型能够学习 QNLI 常见的事实表达和句内关系。加入 3,388 条源任务数据后 Accuracy 下降 0.30 个百分点，Macro-F1 基本不变，说明额外样本数量不能弥补分布差异。",
    )
    add_figure(
        document,
        figures["transfer"],
        "图8  QNLI 上 TF-IDF 迁移设置的 Accuracy 与 Macro-F1。直接迁移 Macro-F1 为 36.72%，目标域 100% 训练达到 58.69%，加入源任务全部数据后的适配训练为 58.69%。源任务并未在完整目标域数据上提供额外提升，说明跨领域词汇和输入结构差异抵消了额外样本的收益。",
        width=5.0,
    )
    efficiency = pd.read_csv(RESULT_STEP3 / "data_efficiency_summary.csv", encoding="utf-8-sig")
    add_caption(document, "表6  数据效率实验（3 个抽样种子的均值与标准差）")
    add_data_table(
        document,
        ["表示", "训练方式", "数据比例", "训练行数", "Accuracy", "Macro-F1"],
        [
            (
                row.representation,
                row.setting,
                row.fraction,
                f"{int(row.train_rows_mean):,}",
                f"{row.accuracy_mean:.2%} ± {row.accuracy_std:.2%}",
                f"{row.macro_f1_mean:.2%} ± {row.macro_f1_std:.2%}",
            )
            for row in efficiency.itertuples()
        ],
        [0.65, 1.25, 0.58, 0.78, 1.0, 1.05],
        font_size=7.7,
        center_columns={0, 2, 3, 4, 5},
    )
    add_figure(
        document,
        figures["efficiency"],
        "图9  QNLI 数据效率学习曲线。MiniLM 在 10% 目标数据时已达到 76.22% Macro-F1，100% 时为 78.60%；TF-IDF 对应为 56.13% 和 58.69%。两种表示在 10% 到 30% 之间的增幅最大，30% 到 100% 的收益明显变小，说明该目标任务的主要可分信息可以在较少数据中获得。",
        width=5.45,
    )
    add_body_paragraph(
        document,
        "适配训练在 10% 和 30% 目标数据时没有超过 target-only，100% 时与 target-only 持平或略低。主要原因是源任务使用长政治采访回答，目标任务是短句对判断；源任务特征虽然仍能提供一定语义信息，但会把模型推向源领域词汇和表达方式。字段接口、标签映射、评价流程可以复用，特征尺度和领域分布必须重新适配。",
    )
    add_body_paragraph(
        document,
        "数据效率结果也说明两个表示的误差来源不同。TF-IDF 随数据量增长缓慢，10% 与 100% 只相差约 2.56 个百分点，说明词面特征在少量数据下已经接近其表示上限；MiniLM 在 10% 时已经较高，后续提升也不大，但整体上限高约 20 个百分点。若标注预算只能覆盖 3,000 条目标样本，MiniLM 的收益更明显；若需要快速建立可解释基线，TF-IDF 仍适合作为第一层模型。",
    )

    add_heading(document, "八、复现与运行记录", 1)
    add_figure(
        document,
        figures["screenshot"],
        "图10  实际终端运行结果截图。图中展示了 QEvasion 七组模型的指标输出与 QNLI 数据效率汇总，数字与项目中的 CSV/JSON 文件一致。复现实验使用固定随机种子，数据下载、字段映射、模型训练和图表生成均由分步 PowerShell 入口执行。",
        width=5.5,
    )
    add_body_paragraph(
        document,
        "项目入口为 run_step1.ps1 至 run_step4.ps1。第二步和第三步分别耗时 253.7 秒和 264.8 秒。第一步的七个确定性产物经过 SHA-256 重跑比对，结果全部一致。完整预测文件包含 gold、prediction、is_error、三个类别概率和 confidence，可直接用于重新生成错误分析。",
    )

    add_heading(document, "九、实验结论", 1)
    add_body_paragraph(
        document,
        "问答回避检测比普通情感分类更依赖问题与回答之间的关系。情感分类通常只需要判断文本自身倾向，而回答清晰度需要判断回答是否覆盖问题所需的信息。一段回答可以同时包含短明确句、背景说明和后续转向，标签往往取决于多个片段共同构成的整体语义。QEvasion 中 Ambivalent 占 59.16%，Clear Non-Reply 只占 10.32%，类别不均衡使多数类基线的 Accuracy 达到 66.88%，但 Macro-F1 只有 26.72%。因此，只看准确率会掩盖少数类识别失败。",
    )
    add_body_paragraph(
        document,
        "传统词级 TF-IDF 在 balanced 设置下取得 40.94% Macro-F1，说明关键词和短语仍是有效信号；不使用类别权重时 Accuracy 更高，但少数类召回下降。MiniLM 拼接基线只有 35.76% Macro-F1，主要限制来自长回答截断。改进模型用首段、中段和尾段的分块表示保留回答结构，并结合问题与回答交互特征，将 Macro-F1 提升到 44.99%，验证了长文本分块对本题的重要性。",
    )
    add_body_paragraph(
        document,
        "QNLI 迁移揭示了复用边界。MiniLM 直接迁移的 Macro-F1 为 66.20%，说明预训练语义表示能在一定程度上跨越领域；TF-IDF 直接迁移只有 36.72%，因为源领域词汇与 QNLI 句对重合有限。目标域训练明显优于直接迁移，但加入全部源任务数据并未进一步提高完整目标域性能。可复用的是 text_a/text_b 接口、标签适配器和统一评价流程；需要修改的是领域特征、长度处理和模型校准。数据效率实验还表明，MiniLM 使用 10% 目标数据即可达到 76.22% Macro-F1，30% 达到 76.97%，100% 只提升 1.64 个百分点，适合在标注预算有限时优先采用强预训练表示。",
    )

    add_heading(document, "十、参考资料", 1)
    references = [
        "[1] Thomas K, Filandrianos G, Lymperaiou M, et al. “I Never Said That”: A dataset, taxonomy and baselines on response clarity classification. Findings of EMNLP 2024: 5204-5233.",
        "[2] Wang A, Singh A, Michael J, et al. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR 2019.",
        "[3] Rajpurkar P, Zhang J, Lopyrev K, et al. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.",
        "[4] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.",
        "[5] Wang W, Wei F, Dong L, et al. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS 2020.",
    ]
    for ref in references:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Pt(0)
        paragraph.paragraph_format.first_line_indent = Pt(-18)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(ref)
        set_run_font(run, "宋体", 9.2, False)


def fill_final_paper_header(document: Document) -> None:
    table = document.tables[0]
    values = {
        (1, 1): STUDENT_ID,
        (1, 6): STUDENT_NAME,
        (1, 9): "2025级",
        (4, 2): "面向对话问答的回答清晰度检测与跨场景迁移及数据效率分析",
    }
    for (row, col), value in values.items():
        cell = table.cell(row, col)
        cell.text = value
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_run_font(run, "宋体", 8.8 if (row, col) == (1, 1) else 10.5, False)


def add_final_paper_body(document: Document, figures: dict[str, Path]) -> None:
    paragraph = document.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("面向对话问答的回答清晰度检测与跨场景迁移及数据效率分析")
    set_run_font(run, "黑体", 18, True)
    run.font.color.rgb = BLACK

    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(f"{STUDENT_NAME}    学号：{STUDENT_ID}")
    set_run_font(run, "宋体", 10.5, False)

    add_heading(document, "摘要", 1)
    add_body_paragraph(
        document,
        "本文研究对话问答场景中的回答清晰度检测，并在 QEvasion 与 QNLI 之间检验统一文本对接口的可迁移性。实验比较多数类、四组 TF-IDF 基线、MiniLM 拼接表示和一种面向长回答的分块交互表示，采用 Macro-F1、类别级指标、混淆矩阵和按采访 URL 分组的交叉验证。改进模型在 QEvasion 官方测试集上取得 51.30% Accuracy 和 44.99% Macro-F1，高于词级 TF-IDF balanced 的 40.94% Macro-F1。QNLI 实验中，MiniLM 直接迁移达到 66.20% Macro-F1，目标域 10% 数据即可达到 76.22%。结果显示，长回答分块和问题-回答交互是提升源任务性能的关键，强预训练表示能降低目标域标注需求，但源领域数据并不必然优于完整目标域训练。",
    )
    paragraph = document.add_paragraph()
    run = paragraph.add_run("关键词：")
    set_run_font(run, "黑体", 10.5, True)
    run = paragraph.add_run("回答清晰度；问答回避；QEvasion；QNLI；跨场景迁移；数据效率")
    set_run_font(run, "宋体", 10.5, False)

    add_heading(document, "一、背景介绍与研究问题", 1)
    add_body_paragraph(
        document,
        "政治访谈、客服对话和安全审核都经常出现“回答了但没有提供所需信息”的情况。回答者可能明确拒绝，也可能用背景、价值立场或相邻议题替代具体答案。此类现象在政治传播研究中被称为含糊或回避，在本文任务中表现为回答清晰度。与情感分类相比，回答清晰度不是从单一文本判断正负，而是判断问题与回答之间是否形成信息上的回应关系。",
    )
    add_body_paragraph(
        document,
        "QEvasion 由 Thomas 等提出，包含政治采访中的问题-回答对，并提供高层清晰度和细粒度回避策略两级标签[1]。高层标签为 Clear Reply、Ambivalent 和 Clear Non-Reply；细粒度标签进一步区分显性、隐性、绕圈、一般化、偏向、部分回答、拒绝、声称无知和澄清。数据集的价值在于把对话中的信息充分性转化为可评价的分类问题，同时保留具体采访来源，便于检查同源泄漏。",
    )
    add_body_paragraph(
        document,
        "QNLI 来自 GLUE 基准中的 Question Natural Language Inference 任务，输入为 question 和 sentence，标签表示句子是否包含问题答案[2,3]。两类任务共享 text_a、text_b 和离散标签的基本结构，但领域、文本长度和标签粒度明显不同。QEvasion 的答案通常是完整的政治采访回答，平均约 1,663 字符；QNLI 的 sentence 通常是 Wikipedia 事实句，平均约 165 字符。",
    )
    add_body_paragraph(
        document,
        "本文提出三个研究问题。RQ1：在类别不均衡的 QEvasion 上，哪一种文本表示能取得更均衡的 Macro-F1？RQ2：从 QEvasion 直接迁移或适配到 QNLI 时，哪些字段和流程可以复用，跨领域边界出现在哪里？RQ3：在 QNLI 目标域上，10%、30% 和 100% 标注数据分别带来多少收益，强预训练表示能否减少标注需求？",
    )
    add_body_paragraph(
        document,
        "类别不均衡和来源泄漏是本研究的两项基本风险。QEvasion 训练集中 Ambivalent 占 59.16%，Clear Non-Reply 只占 10.32%；多数类模型可以获得较高 Accuracy，却无法识别少数类。训练集与官方测试集有 39 个相同采访 URL，对应训练集 590 行、测试集 118 行。若随机切分而不按采访分组，模型可能记住同一采访中的说话风格和议题，导致性能被高估。",
    )

    add_heading(document, "二、相关工作的优缺点总结", 1)
    add_body_paragraph(
        document,
        "Thomas 等构建了 QEvasion 标签体系，并比较传统基线、RoBERTa、DeBERTa、XLNet 和 Llama 等方法[1]。该工作提供了清晰的任务定义和可解释的细粒度类别，但数据主要来自美国政治采访，来源集中，且回答长度远高于常见句分类数据。公开结果说明了预训练模型的上限，却没有解决长回答的分块表示和跨领域标注效率问题。",
    )
    add_body_paragraph(
        document,
        "传统文本分类常使用词袋或 TF-IDF 表示。其优点是训练快、可解释、对少量数据稳定；缺点是依赖词面重合，难以区分否定、转折和跨句信息。GLUE 与 QNLI 提供了统一的自然语言理解评测框架[2,3]。QNLI 的二元标签便于测量迁移，但句子级事实判断与长回答中的信息充分性仍有明显差异。",
    )
    add_body_paragraph(
        document,
        "Sentence-BERT 通过共享编码器生成句子向量，使语义相似度与分类流程更易复用[4]。MiniLM 通过自注意力蒸馏压缩预训练 Transformer，在保持多数能力的同时降低推理成本[5]。这些表示适合 QNLI 的短句对，但直接编码长政治回答时会发生截断。Longformer 等工作通过稀疏注意力处理长文档[7]，其思想说明长文本需要专门结构，而不应把整段回答简单截断。",
    )
    add_body_paragraph(
        document,
        "迁移学习研究通常区分特征复用、参数适配和样本层面适配[6]。本文不微调大型 Transformer，而是在同一 MiniLM 编码器上比较源特征、目标特征和二者拼接。该设计成本低，能够分离接口复用、领域分布差异与训练样本规模的影响。数据效率部分则借用少样本学习的核心问题：在标签预算有限时，强预训练表示是否能比词面模型更早达到可用性能。",
    )

    add_heading(document, "三、模型方法与扩展设计", 1)
    create_method_pipeline(figures["pipeline"])
    add_figure(
        document,
        figures["pipeline"],
        "图1  统一文本对实验流程。QEvasion 与 QNLI 首先转换为相同字段结构，再进入传统特征、预训练表示和统一评价模块；迁移部分固定源标签到目标标签的映射，并对目标域数据比例与抽样种子进行控制。",
        width=6.1,
    )
    add_body_paragraph(
        document,
        "字段适配器把 question 映射到 text_a，把 interview_answer 或 sentence 映射到 text_b，把高层标签映射到 label，把采访 URL 或 president 映射到 group。QEvasion 保留 evasion_label 供 EDA 使用。QNLI 的 entailment 映射为“包含答案”，not_entailment 映射为“不含答案”。",
    )
    add_body_paragraph(
        document,
        "TF-IDF 使用 1-gram 或 1-2 gram 词级特征，min_df=2、max_df=0.98、sublinear_tf=True。逻辑回归 C=2.0，class_weight='balanced'。配对改进使用 MiniLM 编码问题、回答首段、中段和尾段，组合问题向量、回答均值、回答最大值、绝对差、乘积和长度、句数、否定词、转折词、拒答词和模糊词等统计量。分类器 C 通过 GroupKFold 从 0.3、1、3、10 中选择。",
    )
    add_body_paragraph(
        document,
        "QNLI 主实验固定 30,000 条训练池和 2,000 条评估集。数据效率扩展使用 10%、30% 和 100% 三个比例，抽样种子为 42、52、62，报告均值和标准差。100% 设置为确定性子集，重复运行不会改变样本，因此标准差为 0。表示比较包括 TF-IDF 和 MiniLM；MiniLM 只运行 target-only 与 source+target 两组，减少不必要的重复训练。",
    )

    add_heading(document, "四、实验结果与综合分析", 1)
    add_heading(document, "4.1 QEvasion 主任务结果", 2)
    add_caption(document, "表1  QEvasion 模型总体性能（官方 test，n=308）")
    table3 = pd.read_csv(RESULT_STEP2 / "table3_model_comparison.csv", encoding="utf-8-sig")
    add_data_table(
        document,
        ["模型", "Accuracy", "Macro-P", "Macro-R", "Macro-F1"],
        [
            (
                row.model,
                f"{row.accuracy:.2%}",
                f"{row.macro_precision:.2%}",
                f"{row.macro_recall:.2%}",
                f"{row.macro_f1:.2%}",
            )
            for row in table3.itertuples()
        ],
        [2.2, 0.82, 0.82, 0.82, 0.82],
        font_size=7.9,
        center_columns={1, 2, 3, 4},
    )
    add_body_paragraph(
        document,
        "表7 的关键结论不是迁移失败，而是可复用部分的边界清楚。源任务模型直接用于 QNLI 时，词面特征几乎全部失效；目标域训练恢复到 58.69% Macro-F1；适配训练与目标域基线持平，说明源数据只能在目标样本很少时提供有限正则作用。报告因此把字段适配和评价流程视为真正复用，把词表、领域权重和模型决策边界视为必须重训的部分。",
    )
    add_figure(
        document,
        figures["model_comparison"],
        "图2  QEvasion 各模型的 Accuracy 与 Macro-F1 对比。多数类和不加权模型表现出较高的 Accuracy 但较低的 Macro-F1，而改进 MiniLM 的 Macro-F1 最高，更符合类别均衡评价目标。",
        width=6.0,
    )
    add_body_paragraph(
        document,
        "多数类基线的 Accuracy 为 66.88%，Macro-F1 仅 26.72%，说明多数类策略会掩盖少数类失败。TF-IDF balanced 的 Macro-F1 为 40.94%，比对应的无权重配置更均衡。改进 MiniLM 的 Accuracy 为 51.30%，但 Macro-F1 为 44.99%，三类召回率均超过 50%，较适合标签不均衡且关注少数类的任务。",
    )
    add_caption(document, "表2  最佳模型类别级指标")
    table4 = pd.read_csv(RESULT_STEP2 / "table4_best_classification_report.csv", encoding="utf-8-sig")
    add_data_table(
        document,
        ["类别", "Precision", "Recall", "F1", "支持数"],
        [
            (row.label, f"{row.precision:.2%}", f"{row.recall:.2%}", f"{row.f1:.2%}", int(row.support))
            for row in table4.itertuples()
        ],
        [2.15, 0.9, 0.9, 0.9, 0.8],
        center_columns={1, 2, 3, 4},
    )
    add_figure(
        document,
        figures["confusion"],
        "图3  最佳模型混淆矩阵。主要错误集中在 Ambivalent 与 Clear Reply 之间，以及 Ambivalent 与 Clear Non-Reply 之间；Clear Non-Reply 的召回率已超过一半，但精确率仍低，反映拒答与含糊回答的边界。",
        width=5.6,
    )
    add_body_paragraph(
        document,
        "按采访 URL 分组的五折交叉验证中，改进 MiniLM 的平均 Macro-F1 为 49.37%。删除与训练集同源采访后的 190 条测试子集上，Macro-F1 为 48.33%。官方全体 test 的 44.99% 反而低于分组清洁子集，说明泄漏不是性能提升的主要来源；样本减少与不同采访难度会造成波动。",
    )

    add_heading(document, "4.2 五类误判样本", 2)
    add_caption(document, "表3  典型误判样本")
    table5 = pd.read_csv(RESULT_STEP4 / "table5_error_cases.csv", encoding="utf-8-sig")
    add_data_table(
        document,
        ["序号", "问题摘要", "真实", "预测", "错误类型", "原因"],
        [
            (
                index,
                f"{row.question_summary_zh}；{row.answer_summary_zh}",
                row.true_label,
                row.predicted_label,
                row.error_type,
                row.reason_zh,
            )
            for index, row in enumerate(table5.itertuples(), start=1)
        ],
        [0.45, 2.35, 0.8, 0.85, 0.9, 1.25],
        font_size=7.6,
        center_columns={0, 2, 3},
    )
    add_body_paragraph(
        document,
        "第一类错误是长回答先给出短答案，随后转入背景和相邻议题。分块池化会保留尾部信息，但整体分类仍可能把明确回答拉向含糊。第二类错误是“没有标准文件”等保留句被看作明确拒答，而回答实际上继续解释了制度标准。第三类错误是回答从选举问题转向汽油价格，模型把与问题相关但不解决原问题的内容判为明确回答。",
    )
    add_body_paragraph(
        document,
        "拒答与含糊的边界是第五类主要问题。短回答“明年某个时候”提供了部分时间信息，标签为 Ambivalent，但模型判为 Clear Non-Reply。仅靠语义相似度无法稳定区分“承诺了一个模糊时间”和“没有回答问题”，需要在后续模型中引入回答类型、句间立场和问题要件覆盖度。",
    )

    add_heading(document, "4.3 QNLI 迁移结果", 2)
    add_figure(
        document,
        figures["length_shift"],
        "图4  QEvasion 与 QNLI 的 text_b 长度分布。源任务的平均长度约为目标任务的十倍，直接迁移会同时面对领域词表、回答结构和标签粒度变化。",
        width=5.7,
    )
    add_caption(document, "表4  原任务与 QNLI 迁移结果")
    table7 = pd.read_csv(RESULT_STEP3 / "table7_transfer_comparison.csv", encoding="utf-8-sig")
    add_data_table(
        document,
        ["设置", "训练行数", "测试行数", "Accuracy", "Macro-F1", "结论"],
        [
            (row.setting, f"{int(row.train_rows):,}", f"{int(row.test_rows):,}", f"{row.accuracy:.2%}", f"{row.macro_f1:.2%}", row.main_conclusion)
            for row in table7.itertuples()
        ],
        [1.2, 0.7, 0.7, 0.8, 0.8, 1.3],
        font_size=7.8,
        center_columns={1, 2, 3, 4},
    )
    add_body_paragraph(
        document,
        "直接迁移的 TF-IDF Macro-F1 为 36.72%，低于目标域 100% 训练的 58.69%。两者使用同一字段接口和分类器，差异主要来自词汇与句子结构。QNLI 的事实句包含大量实体、地点和时间信息，而 QEvasion 的词表集中在政治人物、政策与采访流程；词级模型无法把这些不同词表映射到同一判断规则。",
    )
    add_figure(
        document,
        figures["transfer"],
        "图5  TF-IDF 在 QNLI 上的迁移对比。直接迁移的 Macro-F1 明显低于目标域训练，说明源领域词面特征不能直接套用；加入完整目标域数据后，源数据没有进一步提高性能。",
        width=5.8,
    )
    add_body_paragraph(
        document,
        "TF-IDF 直接迁移 Accuracy 49.45%、Macro-F1 36.72%，接近二分类中的多数类水平，主要因为政治采访和 Wikipedia 句子的词汇分布差异。MiniLM 直接迁移达到 66.20%，说明预训练语义表示保留了一部分跨领域问题-句子关系。目标域 100% TF-IDF 达到 58.69%，完整目标数据仍是最稳定的设置。",
    )

    add_heading(document, "4.4 数据效率扩展", 2)
    efficiency = pd.read_csv(RESULT_STEP3 / "data_efficiency_summary.csv", encoding="utf-8-sig")
    add_caption(document, "表5  QNLI 数据效率结果（均值 ± 标准差）")
    add_data_table(
        document,
        ["表示", "训练方式", "比例", "训练行数", "Macro-F1"],
        [
            (
                row.representation,
                row.setting,
                row.fraction,
                f"{int(row.train_rows_mean):,}",
                f"{row.macro_f1_mean:.2%} ± {row.macro_f1_std:.2%}",
            )
            for row in efficiency.itertuples()
        ],
        [0.7, 1.3, 0.6, 0.9, 1.25],
        font_size=7.8,
        center_columns={0, 2, 3, 4},
    )
    add_figure(
        document,
        figures["efficiency"],
        "图6  QNLI 数据效率学习曲线。两列分别展示 TF-IDF 与 MiniLM 在不同数据比例下的 Macro-F1；误差棒为 10% 和 30% 三个抽样种子的标准差。",
        width=6.0,
    )
    add_body_paragraph(
        document,
        "MiniLM 在 10% 数据时达到 76.22%，30% 时 76.97%，100% 时 78.60%。从 10% 增加到 30% 只提升 0.74 个百分点，从 30% 增加到 100% 提升 1.63 个百分点。TF-IDF 对应为 56.13%、58.06% 和 58.69%，整体低于 MiniLM 约 20 个百分点。",
    )
    add_body_paragraph(
        document,
        "source+target 在 10% 目标数据时为 72.43%，低于 target-only 的 76.22%；30% 时为 75.56%，仍低于 target-only；100% 时为 77.89%，与 target-only 接近。源数据没有帮助模型更快适应 QNLI，说明领域差异和输入长度差是主导因素。字段适配器和评价流程可以迁移，源任务特征不能直接替代目标域标注。",
    )

    add_heading(document, "4.5 结论适用范围与改进方向", 2)
    add_body_paragraph(
        document,
        "本文结论适用于政治访谈长回答和英文短句事实判断两类场景，不应直接外推到中文客服、医疗问答或安全拒答。MiniLM 直接迁移较高，说明问题与候选句之间的语义关系在预训练空间中存在共享结构；但源数据加完整目标数据不增益，说明样本层面的迁移依赖领域相似度。",
    )
    add_body_paragraph(
        document,
        "后续改进可从三方面进行。第一，使用 Longformer 或分块 Transformer 微调，让模型学习跨片段注意力，而不是对分块向量做均值与最大值。第二，把问题拆成信息要件，评价回答覆盖了哪些要件，从而减少只靠整体语义相似度造成的边界错误。第三，在目标域中使用主动学习选择高信息量样本，验证 10% 数据能否进一步接近完整训练效果。",
    )

    add_heading(document, "五、代码与结果链接", 1)
    add_body_paragraph(
        document,
        "GitHub 地址：https://github.com/chilly-21Chelsea/NLP-Response-Clarity-Transfer。仓库包含 run_all.ps1、requirements.txt、数据源哈希、复现记录、模型对比图、学习曲线、汇总指标和代表性预测文件。",
    )
    add_body_paragraph(
        document,
        "本地复现顺序为 run_step1.ps1、run_step2.ps1、run_step3.ps1、run_step4.ps1。提交包不包含 QEvasion 与 QNLI 的完整原始数据，下载脚本会从固定镜像或官方地址恢复；数据源 SHA-256 与抽样设置保存在 results/step1/data_sources.json。",
    )

    add_heading(document, "六、参考文献", 1)
    references = [
        "[1] Thomas K, Filandrianos G, Lymperaiou M, Zerva C, Stamou G. “I Never Said That”: A dataset, taxonomy and baselines on response clarity classification. Findings of EMNLP 2024: 5204-5233.",
        "[2] Wang A, Singh A, Michael J, Hill F, Levy O, Bowman S R. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR 2019.",
        "[3] Rajpurkar P, Zhang J, Lopyrev K, Liang P. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.",
        "[4] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.",
        "[5] Wang W, Wei F, Dong L, Bao H, Yang N, Zhou M. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS 2020.",
        "[6] Pan S J, Yang Q. A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering, 2010, 22(10): 1345-1359.",
        "[7] Beltagy I, Peters M E, Cohan A. Longformer: The Long-Document Transformer. arXiv:2004.05150, 2020.",
        "[8] Devlin J, Chang M W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.",
    ]
    for ref in references:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Pt(0)
        paragraph.paragraph_format.first_line_indent = Pt(-18)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(ref)
        set_run_font(run, "宋体", 9.2, False)


def build_reports() -> dict[str, str]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    figures = {
        "clarity": RESULT_STEP2 / "figures" / "qevasion_clarity_distribution.png",
        "evasion": RESULT_STEP2 / "figures" / "qevasion_evasion_distribution.png",
        "length": RESULT_STEP2 / "figures" / "qevasion_length_by_label.png",
        "model_comparison": RESULT_STEP4 / "figures" / "qevasion_model_comparison.png",
        "confusion": RESULT_STEP2 / "figures" / "confusion_minilm_chunk_interaction_logreg.png",
        "errors": RESULT_STEP4 / "figures" / "qevasion_error_types.png",
        "length_shift": RESULT_STEP3 / "figures" / "qevasion_qnli_length_shift.png",
        "transfer": RESULT_STEP4 / "figures" / "qnli_transfer_comparison.png",
        "efficiency": RESULT_STEP3 / "figures" / "qnli_data_efficiency.png",
        "screenshot": DOC_WORK / "screenshots" / "run_result_screenshot.png",
        "pipeline": ASSETS / "method_pipeline.png",
    }

    report_ref = TEMPLATES / "实验报告模板.docx"
    report_doc = Document(report_ref)
    style_document(report_doc, body_size=10.5)
    fill_report_cover(report_doc)
    move_body_after(report_doc, report_doc.tables[0]._tbl, remove_anchor=True)
    add_experiment_body(report_doc, figures)
    add_page_number_footer(report_doc)
    report_path = REPORTS / "实验报告_对话问答回避检测与跨场景迁移分析.docx"
    report_doc.save(report_path)

    paper_ref = TEMPLATES / "期末大作业模板.docx"
    paper_doc = Document(paper_ref)
    style_document(paper_doc, body_size=10.5)
    fill_final_paper_header(paper_doc)
    move_body_after(paper_doc, paper_doc.tables[0]._tbl)
    add_final_paper_body(paper_doc, figures)
    add_page_number_footer(paper_doc)
    paper_path = REPORTS / "期末大作业_回答清晰度检测与跨场景迁移及数据效率分析.docx"
    paper_doc.save(paper_path)

    result = {"experiment_report": str(report_path), "final_paper": str(paper_path)}
    write_json(DOC_WORK / "generated_report_paths.json", result)
    return result


def main() -> None:
    configure_logging(PROJECT_ROOT / "logs" / "build_reports.log")
    result = build_reports()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
