from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from cl5g.data.telecomts import LABEL_MAPPING, TAXONOMY_LABELS
from cl5g.evaluation.multiclass import ANOMALY_CLASSES, evaluate_multiclass
from cl5g.models.diagnosis import (
    StringLabelClassifier,
    balanced_sample_weights,
    build_diagnosis_candidates,
    fit_diagnosis,
    predict_anomaly_type,
)


def test_dataset_mapping_covers_each_original_anomaly_category_without_merging() -> None:
    assert tuple(label for label in TAXONOMY_LABELS if label != "Normal") == ANOMALY_CLASSES
    assert set(LABEL_MAPPING.values()) == {"Normal", *ANOMALY_CLASSES}
    for label in ANOMALY_CLASSES:
        assert LABEL_MAPPING[label.casefold()] == label


def test_supported_class_metrics_exclude_classes_without_evaluation_support() -> None:
    labels = np.array(["Jamming", "Jamming", "Antenna Failure"], dtype=object)
    predictions = np.array(["Jamming", "Antenna Failure", "Antenna Failure"], dtype=object)

    metrics = evaluate_multiclass(labels, predictions)

    assert metrics["supported_classes"] == ["Jamming", "Antenna Failure"]
    assert metrics["missing_classes"] == [
        label for label in ANOMALY_CLASSES if label not in {"Antenna Failure", "Jamming"}
    ]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"]["counts"][0][4] == 1
    assert metrics["confusion_matrix"]["counts"][4][4] == 1
    assert metrics["per_class"]["Resource Allocation Bugs"]["support"] == 0
    assert metrics["per_class"]["Resource Allocation Bugs"]["recall"] is None


def test_string_label_classifier_returns_source_class_strings_and_probabilities() -> None:
    features = pd.DataFrame(
        {"f1": [0.0, 0.1, 1.0, 1.1], "f2": [0.0, 0.2, 1.0, 0.9]}
    )
    labels = pd.Series(
        ["Jamming", "Jamming", "Antenna Failure", "Antenna Failure"],
        dtype=object,
    )
    model = StringLabelClassifier(
        RandomForestClassifier(n_estimators=10, random_state=11)
    )

    fit_diagnosis(model, features, labels)
    predictions = predict_anomaly_type(model, features)

    assert set(model.classes_) == {"Antenna Failure", "Jamming"}
    assert predictions.tolist() == labels.tolist()
    assert model.predict_proba(features).shape == (4, 2)


def test_balanced_weights_equalize_total_weight_per_class() -> None:
    labels = np.array(["A", "A", "A", "B"], dtype=object)

    weights = balanced_sample_weights(labels)

    assert weights[:3].sum() == pytest.approx(weights[3])


def test_candidate_builder_includes_weighted_and_unweighted_estimators() -> None:
    labels = np.asarray(list(ANOMALY_CLASSES[:2]) * 2, dtype=object)

    candidates = build_diagnosis_candidates(labels, seed=42)

    assert "random_forest_unweighted" in {candidate.name for candidate in candidates}
    assert "random_forest_balanced" in {candidate.name for candidate in candidates}
    xgboost_candidate = next(
        (candidate for candidate in candidates if candidate.name == "xgboost_balanced"),
        None,
    )
    if xgboost_candidate is not None:
        assert xgboost_candidate.balanced_sample_weight
