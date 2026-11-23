from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cl5g.evaluation.binary import evaluate_binary, write_test_figures
from cl5g.models.detection import (
    ArrayInputClassifier,
    build_detection_candidates,
    threshold_comparison,
)


def test_threshold_selection_optimizes_f2_on_supplied_validation_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    threshold, rows = threshold_comparison(labels, probabilities)
    metrics = evaluate_binary(labels, probabilities, threshold)

    assert threshold == 0.35
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["recall"] == 1.0
    assert metrics["f2"] == pytest.approx(10 / 11)
    assert len(rows) == len(set(probabilities) | {0.0, 1.0})


def test_binary_inputs_reject_non_binary_values_before_integer_conversion() -> None:
    with pytest.raises(ValueError, match="binary"):
        threshold_comparison(np.array([0.2, 1.0]), np.array([0.1, 0.9]))

    with pytest.raises(ValueError, match="both binary classes"):
        build_detection_candidates(np.array([0.8, 1.0]), seed=42)


def test_test_figures_write_distinct_precision_recall_and_roc_plots(
    tmp_path,
) -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    summary = evaluate_binary(labels, probabilities, 0.35)

    write_test_figures(labels, probabilities, 0.35, summary, tmp_path)

    pr_path = tmp_path / "test_precision_recall_curve.png"
    roc_path = tmp_path / "test_roc_curve.png"
    assert pr_path.is_file()
    assert roc_path.is_file()
    assert pr_path.read_bytes() != roc_path.read_bytes()


def test_xgboost_adapter_preserves_feature_order_without_restricted_names() -> None:
    xgboost = pytest.importorskip("xgboost")
    features = np.arange(80, dtype=np.float32).reshape(20, 4)
    frame = pd.DataFrame(
        features,
        columns=["kpi<empty>", "kpi[1]", "kpi__2", "kpi__3"],
    )
    labels = np.array([0, 1] * 10)
    detector = ArrayInputClassifier(
        xgboost.XGBClassifier(
            n_estimators=5,
            max_depth=2,
            n_jobs=1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
    ).fit(frame, labels)

    assert detector.predict_proba(frame).shape == (20, 2)
    with pytest.raises(ValueError, match="Feature names or order"):
        detector.predict_proba(frame.loc[:, list(reversed(frame.columns))])
