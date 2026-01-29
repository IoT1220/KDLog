# %%
import json
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Iterable

from box import Box, BoxList
import einops as ein
import fire
import numpy as np
import pandas as pd
from sklearn import metrics


# %%
def cls_report(y_true, y_pred, *, labels=None, weights=None):
    """
    计算整体和每类的指标

    ## params:

    - labels 真值
    - preds 预测值

    ## returns:

    包含整体和详细指标的字典
    """
    report = metrics.classification_report(
        y_true, y_pred, labels=labels, output_dict=True
    )
    report = Box(report)

    # 由于 accuracy 是一个特殊的指标，我们需要单独处理
    if "accuracy" in report:
        report["micro avg/accuracy"] = {
            "precision": report.accuracy,
            "recall": report.accuracy,
            "f1-score": report.accuracy,
            "support": report["macro avg"].support,
        }
        del report.accuracy
    else:
        report["micro avg/accuracy"] = report["micro avg"]
        del report["micro avg"]

    # 计算加权平均
    if weights is not None:
        report["custom weighted avg"] = {
            key: np.average(
                [report[str(label)][key] for label in labels], weights=weights
            )
            for key in ("precision", "recall", "f1-score")
        }
        report["custom weighted avg"]["support"] = report["macro avg"].support

    report = pd.DataFrame(report).T

    return report


# %%
def states_jsonl_to_metrics_csv(
    # dataset_path: str | Path,
    states_path: str | Path,
    labels: list | None = None,
    weights: list | None = None,
    need_time: bool = True,
):
    if labels is None:
        labels = list(range(4))
    # if weights is None:
    #     weights = [3 / 7, 2 / 7, 1 / 7, 1 / 7]

    # 将路径转换为 Path 对象
    states_path = Path(states_path).expanduser().resolve()
    # dataset_path = states_path.parent

    # 读取states.jsonl
    states = pd.read_json(states_path, lines=True, orient="records")
    y_true, y_pred = states.label.to_numpy(), states.pred.to_numpy()

    # 计算性能指标
    report = cls_report(y_true, y_pred, labels=labels, weights=weights)

    if need_time:
        # 计算效率指标
        times = states[["time", "time_ns"]]
        times.to_csv(states_path.with_suffix(".time.csv"), index=False)
        times.describe().to_csv(states_path.with_name(".time_describe.csv"), index=True)

    # 保存指标
    report.to_csv(states_path.with_suffix(".metrics.csv"))


# %%
if __name__ == "__main__":
    # %%
    # fire.Fire(states_jsonl_to_metrics_csv)

    # %%
    states_jsonl_to_metrics_csv(
        "~/文档/llm-log-check/datasets/compressed/gpt_reason.jsonl",
        labels=[1, 2, 3],
        weights=[5 / 7, 1 / 7, 1 / 7],
        need_time=False,
    )
