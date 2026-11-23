from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from cl5g.data.telecomts import CATEGORICAL_KPIS, KPI_NAMES, NUMERIC_KPIS
from cl5g.data.telecomts_preprocessing import (
    ENCODED_FEATURE_NAMES,
    TelecomTSPreprocessor,
)
from cl5g.features.builder import build_features
from cl5g.features.extraction import (
    BASE_STATISTICS,
    FEATURE_EXPERIMENTS,
    TEMPORAL_STATISTICS,
    build_feature_configuration,
    calculate_statistic,
    extract_features,
    feature_dimension_report,
)
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.pipeline.inference import InferencePipeline


class FeatureParityDetector:
    classes_ = np.array([0, 1])

    def __init__(self, expected: pd.DataFrame) -> None:
        self.expected = expected

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        pd.testing.assert_frame_equal(features, self.expected)
        return np.array([0])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        pd.testing.assert_frame_equal(features, self.expected)
        return np.array([[0.9, 0.1]])


def _raw_sample() -> dict[str, list[float] | list[str]]:
    sample: dict[str, list[float] | list[str]] = {
        name: np.linspace(1.0, 128.0, 128).tolist() for name in NUMERIC_KPIS
    }
    sample[CATEGORICAL_KPIS[0]] = ["TCP" if i % 2 else "" for i in range(128)]
    sample[CATEGORICAL_KPIS[1]] = ["UDP" if i % 3 else "None" for i in range(128)]
    return sample


def test_experiment_feature_dimensions_and_order() -> None:
    report = feature_dimension_report()
    expected_dimensions = {
        "base_statistics": 122,
        "base_plus_temporal": 250,
        "base_plus_distribution": 170,
        "base_plus_mean_absolute_difference": 138,
        "selected_trend": 170,
        "selected_shape": 170,
        "selected_temporal_core": 202,
    }

    assert set(report["feature_sets"]) == set(FEATURE_EXPERIMENTS)
    for feature_set, dimension in expected_dimensions.items():
        configuration = build_feature_configuration(feature_set)
        description = report["feature_sets"][feature_set]
        assert description["total_feature_count"] == dimension
        assert description["feature_order"] == configuration["feature_order"]
        assert len(set(configuration["feature_order"])) == dimension


def test_base_and_temporal_statistics_match_reference_values() -> None:
    sequence = np.arange(128, dtype=np.float64)
    configuration = build_feature_configuration("base_plus_temporal")
    values = TelecomTSPreprocessor().transform_sample({"kpis": _raw_sample()})
    values[:, 0] = sequence

    features = extract_features(values, configuration).iloc[0]

    assert tuple(BASE_STATISTICS) == (
        "mean",
        "std",
        "min",
        "max",
        "median",
        "first",
        "last",
    )
    assert tuple(TEMPORAL_STATISTICS) == (
        "slope",
        "first_half_mean",
        "second_half_mean",
        "mean_absolute_difference",
        "root_mean_square",
        "skewness",
        "excess_kurtosis",
        "interquartile_range",
    )
    assert features["RSRP__mean"] == pytest.approx(63.5)
    assert features["RSRP__slope"] == pytest.approx(1.0)
    assert features["RSRP__first_half_mean"] == pytest.approx(31.5)
    assert features["RSRP__second_half_mean"] == pytest.approx(95.5)
    assert features["RSRP__mean_absolute_difference"] == pytest.approx(1.0)
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_constant_and_large_sequences_produce_finite_features() -> None:
    configuration = build_feature_configuration("base_plus_temporal")
    values = TelecomTSPreprocessor().transform_sample({"kpis": _raw_sample()})
    values[:, 0] = 1e300

    features = extract_features(values, configuration).iloc[0]

    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert features["RSRP__mean"] == pytest.approx(1e300)
    assert features["RSRP__std"] == 0.0
    assert features["RSRP__slope"] == 0.0
    assert features["RSRP__skewness"] == 0.0
    assert features["RSRP__excess_kurtosis"] == 0.0
    assert calculate_statistic(np.full(128, 1e300), "root_mean_square") == pytest.approx(
        1e300
    )


def test_category_frequency_features_match_one_hot_protocols() -> None:
    sample = _raw_sample()
    configuration = build_feature_configuration("base_statistics")
    encoded = TelecomTSPreprocessor().transform_sample({"kpis": sample})

    features = extract_features(encoded, configuration).iloc[0]

    assert features["UL_Protocol__frequency__<empty>"] == pytest.approx(0.5)
    assert features["UL_Protocol__frequency__TCP"] == pytest.approx(0.5)
    assert features["DL_Protocol__frequency__UDP"] == pytest.approx(85 / 128)
    assert features["DL_Protocol__frequency__None"] == pytest.approx(43 / 128)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_missing_and_non_finite_processed_values_are_rejected(
    invalid_value: float,
) -> None:
    configuration = build_feature_configuration("base_statistics")
    values = TelecomTSPreprocessor().transform_sample({"kpis": _raw_sample()})
    values[0, 0] = invalid_value

    with pytest.raises(ValueError, match="non-finite"):
        extract_features(values, configuration)


def test_shape_one_hot_and_repeated_extraction_are_validated() -> None:
    configuration = build_feature_configuration("base_statistics")
    values = TelecomTSPreprocessor().transform_sample({"kpis": _raw_sample()})

    with pytest.raises(ValueError, match="shape"):
        extract_features(values[:-1], configuration)

    invalid_one_hot = values.copy()
    invalid_one_hot[0, 16:21] = 0.0
    with pytest.raises(ValueError, match="one-hot"):
        extract_features(invalid_one_hot, configuration)

    first = extract_features(values, configuration)
    second = extract_features(values.copy(), configuration)
    pd.testing.assert_frame_equal(first, second)


def test_invalid_category_configuration_is_reported_as_value_error() -> None:
    configuration = build_feature_configuration("base_statistics")
    category_feature = next(
        feature
        for feature in configuration["features"]
        if feature["statistic"] == "category_frequency"
    )
    category_feature["source_kpi"] = []

    with pytest.raises(ValueError, match="source_kpi"):
        extract_features(
            np.zeros((128, len(ENCODED_FEATURE_NAMES))),
            configuration,
        )


def test_training_builder_and_inference_use_identical_features(tmp_path: Path) -> None:
    sample = _raw_sample()
    configuration = build_feature_configuration("base_plus_temporal")
    training_frame = pd.DataFrame([{"kpis": sample}])
    expected = build_features(training_frame, configuration)

    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    base = load_config()
    inference = replace(
        base.inference,
        detection_artifact="artifacts/detection.joblib",
        preprocessing_artifact="artifacts/preprocessing.joblib",
        feature_config_artifact="artifacts/feature_config.json",
        metadata_artifact="artifacts/model_metadata.json",
        attribution_artifact=None,
        temporal_localization_artifact=None,
    )
    config: AppConfig = replace(
        base,
        project_root=tmp_path,
        inference=inference,
    )
    joblib.dump(FeatureParityDetector(expected), artifact_directory / "detection.joblib")
    joblib.dump(TelecomTSPreprocessor(), artifact_directory / "preprocessing.joblib")
    (artifact_directory / "feature_config.json").write_text(
        json.dumps(configuration),
        encoding="utf-8",
    )
    (artifact_directory / "model_metadata.json").write_text(
        json.dumps({"model_version": "feature-parity-test"}),
        encoding="utf-8",
    )
    matrix = [
        [sample[kpi][step] for kpi in KPI_NAMES]
        for step in range(128)
    ]

    prediction = InferencePipeline(config).predict_binary(matrix)

    assert prediction["status"] == "normal"
    assert prediction["model_version"] == "feature-parity-test"
