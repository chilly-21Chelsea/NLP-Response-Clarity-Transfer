from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_pipeline.common import PROJECT_ROOT, configure_logging, write_json
from final_pipeline.experiment_utils import configure_plots, paths_for_step


SELECTED_ERRORS = {
    245: {
        "question_summary": "美国在中东是否因信誉问题而受到限制？",
        "answer_summary": "回答先否认存在信誉担忧，随后转向黎巴嫩驻军、后勤支援和达尔富尔等背景。",
        "error_type": "长而空泛",
        "reason": "开头含有短否定句，但后续大量背景铺陈使回答是否真正解决问题变得模糊。",
    },
    267: {
        "question_summary": "是否正在考虑设置禁飞区？",
        "answer_summary": "回答先说禁飞区是正在考虑的一个选项，随后扩展到利比亚、墨西哥和其他外交议题。",
        "error_type": "关键短句被长回答淹没",
        "reason": "明确回答只出现在前段，后续大段延伸使模型把整段判为含糊回答。",
    },
    157: {
        "question_summary": "是否认为自己在使用赦免权时过于吝啬？",
        "answer_summary": "回答称没有把标准文件带在身边，但随后解释司法部和白宫法律顾问有严格标准。",
        "error_type": "部分回答",
        "reason": "回答既包含无法直接比较的保留，也包含制度性解释，介于含糊回答与明确拒答之间。",
    },
    188: {
        "question_summary": "为什么这项安排是好主意或坏主意？",
        "answer_summary": "回答先回避被拉入选举话题，随后转向汽油价格和白宫关注的问题。",
        "error_type": "主题转移",
        "reason": "回答没有处理原问题中的判断理由，而是切换到另一组政策和政治议题。",
    },
    212: {
        "question_summary": "事情会在接下来几个月发生吗？",
        "answer_summary": "回答仅给出“明年某个时候”这一粗略时间，没有确认具体月份或条件。",
        "error_type": "标签边界模糊",
        "reason": "短回答提供了部分时间信息，但不足以构成完整明确回答，因此人工标签与模型判断存在边界冲突。",
    },
}


def truncate(text: object, limit: int) -> str:
    value = "" if pd.isna(text) else str(text).replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def model_comparison_chart(table3: pd.DataFrame, path: Path) -> None:
    display = {
        "majority_baseline": "Majority",
        "tfidf_uni_lr_none": "TF-IDF uni LR",
        "tfidf_uni_lr_balanced": "TF-IDF uni balanced",
        "tfidf_uni_bi_lr_none": "TF-IDF uni+bi LR",
        "tfidf_uni_bi_lr_balanced": "TF-IDF uni+bi balanced",
        "minilm_concat_logreg": "MiniLM concat",
        "minilm_chunk_interaction_logreg": "MiniLM chunk+features",
    }
    frame = table3.copy()
    frame["display"] = frame["model"].map(display).fillna(frame["model"])
    frame = frame.sort_values("macro_f1", ascending=True)
    y = np.arange(len(frame))
    plt.figure(figsize=(9.2, 5.2))
    plt.barh(y - 0.18, frame["accuracy"], height=0.36, label="Accuracy")
    plt.barh(y + 0.18, frame["macro_f1"], height=0.36, label="Macro-F1")
    plt.yticks(y, frame["display"])
    plt.xlim(0, 0.8)
    plt.xlabel("score")
    plt.title("QEvasion model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def transfer_comparison_chart(table7: pd.DataFrame, path: Path) -> None:
    display = {
        "original_qevasion_best": "QEvasion best",
        "direct_transfer": "Direct transfer",
        "target_baseline": "QNLI baseline",
        "adapted_training": "Source + QNLI",
    }
    frame = table7.loc[table7["setting"].ne("original_qevasion_best")].copy()
    frame["display"] = frame["setting"].map(display)
    x = np.arange(len(frame))
    width = 0.36
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x - width / 2, frame["accuracy"], width, label="Accuracy")
    plt.bar(x + width / 2, frame["macro_f1"], width, label="Macro-F1")
    plt.xticks(x, frame["display"])
    plt.ylim(0, 0.75)
    plt.ylabel("score")
    plt.title("QNLI transfer comparison with TF-IDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    configure_logging(PROJECT_ROOT / "logs" / "step4_error_and_figures.log")
    configure_plots()
    step2 = PROJECT_ROOT / "results" / "step2"
    step3 = PROJECT_ROOT / "results" / "step3"
    paths = paths_for_step("step4")

    predictions = pd.read_csv(
        step2 / "predictions" / "predictions_best_qevasion.csv",
        encoding="utf-8-sig",
    )
    rows_out = []
    for row_id, annotation in SELECTED_ERRORS.items():
        matches = predictions.loc[predictions["row_id"].eq(row_id)]
        if matches.empty:
            raise ValueError(f"Selected error row {row_id} was not found")
        row = matches.iloc[0]
        if not bool(row["is_error"]):
            raise ValueError(f"Selected row {row_id} is not an error")
        rows_out.append(
            {
                "row_id": row_id,
                "question_summary_zh": annotation["question_summary"],
                "answer_summary_zh": annotation["answer_summary"],
                "true_label": row["gold"],
                "predicted_label": row["prediction"],
                "confidence": float(row["confidence"]),
                "error_type": annotation["error_type"],
                "reason_zh": annotation["reason"],
                "question_excerpt": truncate(row["text_a"], 220),
                "answer_excerpt": truncate(row["text_b"], 520),
                "url": row["url"],
            }
        )
    table5 = pd.DataFrame(rows_out)
    table5.to_csv(
        paths["root"] / "table5_error_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    category_map = {
        "Ambivalent->Clear Reply": "长而空泛/隐式回答",
        "Clear Reply->Ambivalent": "关键短句被长回答淹没",
        "Ambivalent->Clear Non-Reply": "部分回答/拒答边界",
        "Clear Non-Reply->Clear Reply": "主题转移",
        "Clear Non-Reply->Ambivalent": "标签边界模糊",
        "Clear Reply->Clear Non-Reply": "明确拒答误判",
    }
    errors = predictions.loc[predictions["is_error"]].copy()
    errors["pair"] = errors["gold"] + "->" + errors["prediction"]
    errors["error_type"] = errors["pair"].map(category_map).fillna("其他")
    counts = (
        errors.groupby(["error_type", "pair"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    counts.to_csv(
        paths["root"] / "error_type_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plt.figure(figsize=(8.6, 4.8))
    sns.barplot(data=counts, x="count", y="error_type", hue="pair", dodge=False)
    plt.title("QEvasion error types")
    plt.xlabel("count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "qevasion_error_types.png", dpi=180, bbox_inches="tight")
    plt.close()

    table3 = pd.read_csv(step2 / "table3_model_comparison.csv", encoding="utf-8-sig")
    table7 = pd.read_csv(step3 / "table7_transfer_comparison.csv", encoding="utf-8-sig")
    efficiency = pd.read_csv(step3 / "data_efficiency_summary.csv", encoding="utf-8-sig")
    model_comparison_chart(table3, paths["figures"] / "qevasion_model_comparison.png")
    transfer_comparison_chart(table7, paths["figures"] / "qnli_transfer_comparison.png")

    round_trip = {
        "qevasion": {
            "metrics": json.loads((step2 / "metrics_summary.json").read_text(encoding="utf-8")),
            "table3": table3.to_dict(orient="records"),
        },
        "transfer": {
            "metrics": json.loads((step3 / "metrics_summary.json").read_text(encoding="utf-8")),
            "table7": table7.to_dict(orient="records"),
            "efficiency": efficiency.to_dict(orient="records"),
        },
        "errors": {
            "selected_cases": table5.to_dict(orient="records"),
            "type_counts": counts.to_dict(orient="records"),
        },
    }
    write_json(PROJECT_ROOT / "results" / "final_metrics.json", round_trip)
    print(table5.to_string(index=False))


if __name__ == "__main__":
    main()
