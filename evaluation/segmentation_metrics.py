from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class SegmentationMetrics:
    num_classes: int = 23
    ignore_index: int = 0

    def __post_init__(self):
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        self.confusion.fill(0)

    def update(self, target, prediction) -> None:
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)

        if target.shape != prediction.shape:
            raise ValueError(
                f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}"
            )

        valid = (
            (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
            & (target != self.ignore_index)
        )
        target = target[valid]
        prediction = prediction[valid]

        flat = target * self.num_classes + prediction
        hist = np.bincount(
            flat, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def compute(self) -> Dict:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        gt = cm.sum(axis=1)
        pred = cm.sum(axis=0)
        union = gt + pred - tp

        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        present = union > 0
        iou[present] = tp[present] / union[present]

        class_acc = np.full(self.num_classes, np.nan, dtype=np.float64)
        gt_present = gt > 0
        class_acc[gt_present] = tp[gt_present] / gt[gt_present]

        eval_mask = present.copy()
        if 0 <= self.ignore_index < self.num_classes:
            eval_mask[self.ignore_index] = False

        miou = float(np.nanmean(iou[eval_mask])) if eval_mask.any() else float("nan")
        mean_acc = (
            float(np.nanmean(class_acc[eval_mask])) if eval_mask.any() else float("nan")
        )
        overall = float(tp.sum() / cm.sum()) if cm.sum() > 0 else float("nan")

        return {
            "miou": miou,
            "mean_class_accuracy": mean_acc,
            "overall_accuracy": overall,
            "per_class_iou": iou.tolist(),
            "per_class_accuracy": class_acc.tolist(),
            "support": gt.astype(np.int64).tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }
