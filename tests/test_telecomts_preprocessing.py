from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pytest

from cl5g.data.telecomts import CATEGORICAL_KPIS, KPI_NAMES, NUMERIC_KPIS
from cl5g.data.telecomts_preprocessing import (
    ENCODED_FEATURE_NAMES,
    LEAKAGE_FIELDS,
    PROTOCOL_CATEGORIES,
    TelecomTSPreprocessor,
    preprocessing_configuration,
    write_preprocessing_configuration,
)


def _sample() -> dict[str, Any]:
    kpis: dict[str, list[Any]] = {
        name: [float(index) for index in range(128)] for name in NUMERIC_KPIS
    }
    for name in CATEGORICAL_KPIS:
        kpis[name] = ["TCP"] * 128
    kpis["UL_Protocol"][:5] = ["", "None", "TCP", "UDP", None]
    return {
        "sample_id": "sample-1",
        "kpis": kpis,
        "label": "Jamming",
        "binary_label": True,
        "description": "must not enter predictive input",
        "labels": {"zone": "A", "application": "File"},
        "anomalies": {
            "type": "Jamming",
            "anomaly_duration": {"start": 10, "end": 20},
            "affected_kpis": ["RSRP"],
        },
    }


def test_one_hot_encoding_preserves_protocol_categories_and_numeric_values() -> None:
    encoded = TelecomTSPreprocessor().transform_sample(_sample())

    assert encoded.shape == (128, 26)
    assert tuple(ENCODED_FEATURE_NAMES[:16]) == NUMERIC_KPIS
    assert encoded[4, :16].tolist() == [4.0] * 16
    assert encoded[0, 16:21].tolist() == [1, 0, 0, 0, 0]
    assert encoded[1, 16:21].tolist() == [0, 1, 0, 0, 0]
    assert encoded[2, 16:21].tolist() == [0, 0, 1, 0, 0]
    assert encoded[3, 16:21].tolist() == [0, 0, 0, 1, 0]
    assert encoded[4, 16:21].tolist() == [0, 0, 0, 0, 1]
    assert encoded[0, 21:26].tolist() == [0, 0, 1, 0, 0]


def test_metadata_changes_do_not_change_predictive_inputs() -> None:
    preprocessor = TelecomTSPreprocessor()
    sample = _sample()
    first = preprocessor.transform_sample(sample)
    sample["label"] = "Normal"
    sample["binary_label"] = False
    sample["description"] = "different description"
    sample["labels"] = {"zone": "C", "application": "YouTube"}
    sample["anomalies"] = {"affected_kpis": [], "anomaly_duration": None}

    np.testing.assert_array_equal(preprocessor.transform_sample(sample), first)
    assert set(LEAKAGE_FIELDS).issuperset(
        {"label", "anomaly_type", "zone", "application", "mobility", "congestion"}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda sample: sample["kpis"]["RSRP"].__setitem__(0, None), "is missing"),
        (lambda sample: sample["kpis"]["UL_SNR"].__setitem__(0, float("inf")), "not finite"),
        (lambda sample: sample["kpis"]["DL_Protocol"].__setitem__(0, "SCTP"), "unsupported category"),
        (lambda sample: sample["kpis"].__setitem__("Unexpected_KPI", [0.0] * 128), "unexpected KPI"),
        (lambda sample: sample["kpis"]["RSRP"].pop(), "must contain 128 values"),
    ],
)
def test_invalid_values_and_schema_are_rejected(mutation: Any, message: str) -> None:
    sample = _sample()
    mutation(sample)

    with pytest.raises(ValueError, match=message):
        TelecomTSPreprocessor().transform_sample(sample)


def test_non_string_protocol_value_is_rejected() -> None:
    sample = _sample()
    sample["kpis"]["UL_Protocol"][0] = 3

    with pytest.raises(ValueError, match="must be a string or missing"):
        TelecomTSPreprocessor().transform_sample(sample)


def test_training_and_inference_share_the_serialized_encoder(tmp_path) -> None:
    training_transformer = TelecomTSPreprocessor()
    artifact = tmp_path / "preprocessor.joblib"
    joblib.dump(training_transformer, artifact)
    inference_transformer = joblib.load(artifact)
    sample = _sample()

    np.testing.assert_array_equal(
        inference_transformer.transform_sample(sample),
        training_transformer.transform_sample(sample),
    )


def test_preprocessing_configuration_is_reproducible_and_persisted(tmp_path) -> None:
    path = tmp_path / "preprocessing_config.json"
    configuration = write_preprocessing_configuration(
        path,
        revision="a" * 40,
        sample_count=12,
    )

    assert configuration == preprocessing_configuration(
        revision="a" * 40,
        sample_count=12,
    )
    assert configuration["output"]["shape"] == [128, 26]
    assert configuration["categorical_handling"]["categories_per_kpi"] == list(
        PROTOCOL_CATEGORIES
    )
    assert configuration["input"]["kpi_order"] == list(KPI_NAMES)
    assert configuration["fitted_transformations"] == []
    assert json.loads(path.read_text(encoding="utf-8"))["dataset_revision"] == "a" * 40
