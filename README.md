# 对话问答回答清晰度检测与跨场景迁移

本项目完成自然语言处理课程的两份大作业提交物：

- QEvasion 回答清晰度检测实验及其实验报告。
- 在此基础上增加 QNLI 跨场景迁移与数据效率实验的研究论文。

GitHub 仓库：https://github.com/chilly-21Chelsea/NLP-Response-Clarity-Transfer

## 主要结果

QEvasion 官方测试集共 308 条。多数类基线 Accuracy 为 66.88%，Macro-F1 为 26.72%。词级 TF-IDF balanced 基线 Macro-F1 为 40.94%。改进的长回答 MiniLM 分块交互模型取得 51.30% Accuracy 和 44.99% Macro-F1，是七组模型中 Macro-F1 最高的配置。

QNLI 迁移使用固定 30,000 条训练池和 2,000 条评估集。TF-IDF 直接迁移 Macro-F1 为 36.72%，目标域 100% 训练为 58.69%。MiniLM 直接迁移达到 66.20%；仅使用 10% 目标域训练池即达到 76.22%，100% 训练池为 78.60%。

## 数据

- QEvasion：`ailsntua/QEvasion`，训练集 3,448 条、官方测试集 308 条。
- QNLI：`nyu-mll/glue` 的 `qnli` 配置，完整训练集 104,743 条、验证集 5,463 条。
- 固定实验样本：QNLI 训练池 30,000 条，评估集 2,000 条，分层抽样种子为 42。

原始数据不提交到仓库。`run_download_qevasion.ps1` 恢复 QEvasion，第一步脚本从 Hugging Face 镜像恢复 QNLI，并以内置官方 GLUE 压缩包作为兜底。

## 环境

项目在以下环境中完成：

- Python 3.13.15
- Windows 10
- scikit-learn 1.9.0
- sentence-transformers 6.1.0
- PyTorch 2.14.1 CPU

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 复现

传入项目所用 Python 解释器：

```powershell
.\run_all.ps1 -PythonExe "C:\path\to\python.exe"
```

也可以分步执行：

```powershell
.\run_download_qevasion.ps1
.\run_step1.ps1
.\run_step2.ps1
.\run_step3.ps1
.\run_step4.ps1
```

第一步的确定性产物可执行双重运行校验：

```powershell
.\verify_step1.ps1
```

报告构建入口位于 `src/final_pipeline/build_reports.py`。Word 模板保存在 `templates/`。

## 目录

```text
configs/            实验配置
data/               数据说明；原始和中间数据不提交
src/final_pipeline/ 下载、审计、训练、迁移、错误分析、报告生成代码
results/            汇总指标、类别报告、代表性预测、图表和复现记录
docs/reports/       最终实验报告与期末大作业论文
assets/             报告使用的运行截图与流程图
artifact_contracts/ 两份 Word 模板的结构合同
templates/          课程提供的 Word 模板
```

## 提交报告

- [实验报告](docs/reports/实验报告_对话问答回避检测与跨场景迁移分析.docx)
- [期末大作业论文](docs/reports/期末大作业_回答清晰度检测与跨场景迁移及数据效率分析.docx)

运行日志、下载哈希、预测概率和重跑一致性记录均保留在对应结果目录中。
