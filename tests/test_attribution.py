from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from cl5g.models.attribution import KPIAttributionExplainer
from cl5g.models.diagnosis import StringLabelClassifier


@pytest.fixture
def feature_configuration() -> dict[str, object]:
    return {
        "kpi_names": ["RSRP", "DL_BLER"],
        "feature_order": [
            "RSRP__mean",
            "RSRP__std",
            "DL_BLER__mean",
        ],
        "features": [
            {"name": "RSRP__mean", "kpi": "RSRP", "source_kpi": "RSRP"},
            {"name": "RSRP__std", "kpi": "RSRP", "source_kpi": "RSRP"},
            {"name": "DL_BLER__mean", "kpi": "DL_BLER", "source_kpi": "DL_BLER"},
        ],
    }


def test_kpi_aggregation_sums_absolute_and_signed_feature_contributions(
    feature_configuration: dict[str, object],
) -> None:
    explainer = KPIAttributionExplainer(feature_configuration)

    scores = explainer.aggregate_by_kpi(np.array([0.3, -0.2, 0.1]))

    assert scores["RSRP"] == {
        "importance": pytest.approx(0.5),
        "signed_contribution": pytest.approx(0.1),
    }
    assert scores["DL_BLER"] == {
        "importance": pytest.approx(0.1),
        "signed_contribution": pytest.approx(0.1),
    }


def test_tree_shap_explanation_maps_features_to_kpis(
    feature_configuration: dict[str, object],
) -> None:
    training = pd.DataFrame(
        {
            "RSRP__mean": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
            "RSRP__std": [0.0, 0.1, 0.8, 0.9, 0.2, 1.0],
            "DL_BLER__mean": [0.0, 0.2, 0.7, 1.0, 0.1, 0.8],
        }
    )
    labels = np.array([0, 0, 1, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=12, random_state=4).fit(
        training,
        labels,
    )
    sample = training.iloc[[2]]
    explainer = KPIAttributionExplainer(feature_configuration)

    result = explainer.explain_prediction(
        sample,
        model=model,
        model_kind="binary_detection",
        predicted_class=1,
        probability=float(model.predict_proba(sample)[0, 1]),
    )

    assert result["method"] == "TreeSHAP"
    assert result["model_kind"] == "binary_detection"
    assert result["predicted_class"] == 1
    assert result["output_value"] == pytest.approx(result["prediction_probability"])
    assert result["top_features"][0]["feature"] in feature_configuration["feature_order"]
    assert {item["kpi"] for item in result["top_kpis"]} <= {"RSRP", "DL_BLER"}
    assert "not establish physical causality" in result["interpretation"]


def test_tree_shap_selects_string_diagnosis_class(
    feature_configuration: dict[str, object],
) -> None:
    training = pd.DataFrame(
        {
            "RSRP__mean": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
            "RSRP__std": [0.0, 0.1, 0.8, 0.9, 0.2, 1.0],
            "DL_BLER__mean": [0.0, 0.2, 0.7, 1.0, 0.1, 0.8],
        }
    )
    labels = np.array(
        ["Jamming", "Jamming", "Antenna Failure", "Antenna Failure", "Jamming", "Antenna Failure"]
    )
    model = StringLabelClassifier(
        RandomForestClassifier(n_estimators=12, random_state=4)
    ).fit(training, labels)
    sample = training.iloc[[2]]
    predicted = model.predict(sample)[0]
    explainer = KPIAttributionExplainer(feature_configuration)

    result = explainer.explain_prediction(
        sample,
        model=model,
        model_kind="multiclass_diagnosis",
        predicted_class=predicted,
        probability=float(model.predict_proba(sample)[0, list(model.classes_).index(predicted)]),
    )

    assert result["predicted_class"] == predicted
    assert result["output_value"] == pytest.approx(result["prediction_probability"])


def test_explainer_rejects_misaligned_feature_order(
    feature_configuration: dict[str, object],
) -> None:
    model = RandomForestClassifier(n_estimators=2, random_state=1).fit(
        np.array([[0, 0, 0], [1, 1, 1]]),
        [0, 1],
    )
    features = pd.DataFrame([[0, 0, 0]], columns=["wrong", "order", "here"])

    with pytest.raises(ValueError, match="Feature names or order"):
        KPIAttributionExplainer(feature_configuration).explain_prediction(
            features,
            model=model,
            model_kind="binary_detection",
            predicted_class=1,
            probability=0.8,
        )
