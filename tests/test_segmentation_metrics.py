from __future__ import annotations

import numpy as np

from evaluation.segmentation_metrics import SegmentationMetrics


def main():
    m = SegmentationMetrics(num_classes=4, ignore_index=0)
    target = np.array([0, 1, 1, 2, 2, 3])
    pred = np.array([3, 1, 2, 2, 2, 3])
    m.update(target, pred)
    result = m.compute()

    assert result["support"][0] == 0
    assert result["support"][1] == 2
    assert result["support"][2] == 2
    assert result["support"][3] == 1

    iou = result["per_class_iou"]
    assert abs(iou[1] - 0.5) < 1e-12
    assert abs(iou[2] - (2/3)) < 1e-12
    assert abs(iou[3] - 1.0) < 1e-12

    expected_miou = (0.5 + 2/3 + 1.0) / 3
    assert abs(result["miou"] - expected_miou) < 1e-12

    print("PASS: test_segmentation_metrics")


if __name__ == "__main__":
    main()
