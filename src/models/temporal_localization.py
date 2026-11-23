"""Supervised timestep-to-interval localization for TelecomTS sequences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cl5g.data.telecomts import KPI_NAMES, TIME_STEPS
from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor


class TemporalIntervalLocalizer:
    """Turn per-timestep positive-class scores into one inclusive interval."""

    input_contract = "raw_kpi_sequence_v1"
    method = "experimental_supervised_per_timestep"

    def __init__(
        self,
        estimator: object,
        *,
        threshold: float,
        context_radius: int = 2,
    ) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("threshold must be finite and between 0 and 1.")
        if (
            isinstance(context_radius, bool)
            or not isinstance(context_radius, (int, np.integer))
            or context_radius < 0
        ):
            raise ValueError("context_radius must be a non-negative integer.")
        self.estimator = estimator
        self.threshold = float(threshold)
        self.context_radius = int(context_radius)
        self.preprocessor = TelecomTSPreprocessor()

    def predict_scores(self, sample: object) -> np.ndarray:
        """Return one anomaly-interval probability per original time step."""
        encoded = self.preprocessor.transform_sample(_as_kpi_mapping(sample))
        features = temporal_context_features(
            encoded,
            context_radius=self.context_radius,
        )
        predict_proba = getattr(self.estimator, "predict_proba", None)
        classes = getattr(self.estimator, "classes_", None)
        if not callable(predict_proba) or classes is None:
            raise ValueError("The timestep estimator must expose predict_proba and classes_.")
        positive_indexes = [
            index
            for index, value in enumerate(classes)
            if value == 1 or str(value).strip().lower() in {"1", "true", "anomaly"}
        ]
        if len(positive_indexes) != 1:
            raise ValueError("The timestep estimator must have one positive class.")
        probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        if probabilities.shape != (TIME_STEPS, len(classes)):
            raise ValueError("The timestep estimator returned incompatible probabilities.")
        scores = probabilities[:, positive_indexes[0]]
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("The timestep estimator returned invalid probabilities.")
        return scores

    def predict_interval(
        self,
        sample: object,
        *,
        sampling_rate_hz: float = 10.0,
    ) -> dict[str, Any] | None:
        """Predict an interval from the original 128-by-18 KPI sequence."""
        rate = _validate_sampling_rate(sampling_rate_hz)
        scores = self.predict_scores(sample)
        interval = interval_from_scores(scores, self.threshold)
        if interval is None:
            return None
        start, end = interval
        return {
            "start": start,
            "end": end,
            "start_seconds": start / rate,
            "end_seconds": end / rate,
            "duration_seconds": (end - start + 1) / rate,
            "sampling_rate_hz": rate,
            "localization_score": float(np.mean(scores[start : end + 1])),
            "score_calibrated": False,
            "method": self.method,
            "affected_kpis": None,
        }


def temporal_context_features(
    sequence: np.ndarray,
    *,
    context_radius: int = 2,
) -> np.ndarray:
    """Concatenate a centered, edge-padded timestep context window."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != TIME_STEPS or values.shape[1] == 0:
        raise ValueError(
            f"sequence must have shape ({TIME_STEPS}, channels) with at least one channel."
        )
    if not np.isfinite(values).all():
        raise ValueError("sequence values must be finite.")
    if (
        isinstance(context_radius, bool)
        or not isinstance(context_radius, (int, np.integer))
        or context_radius < 0
    ):
        raise ValueError("context_radius must be a non-negative integer.")
    padded = np.pad(
        values,
        ((context_radius, context_radius), (0, 0)),
        mode="edge",
    )
    return np.concatenate(
        [
            padded[offset : offset + TIME_STEPS]
            for offset in range(2 * context_radius + 1)
        ],
        axis=1,
    )


def interval_from_scores(
    scores: np.ndarray,
    threshold: float,
) -> tuple[int, int] | None:
    """Convert timestep scores to a single inclusive first-to-last interval."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (TIME_STEPS,) or not np.isfinite(values).all():
        raise ValueError(f"scores must contain {TIME_STEPS} finite timestep values.")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be finite and between 0 and 1.")
    positive = np.flatnonzero(values >= threshold)
    if positive.size == 0:
        return None
    return int(positive[0]), int(positive[-1])


def _as_kpi_mapping(sample: object) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        kpis = sample.get("kpis", sample)
        if isinstance(kpis, Mapping):
            return kpis
        raise ValueError("sample must contain a KPI mapping.")
    try:
        values = np.asarray(sample, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample must be a 128-by-18 KPI sequence.") from exc
    if values.shape != (TIME_STEPS, len(KPI_NAMES)):
        raise ValueError(
            f"sample must have shape ({TIME_STEPS}, {len(KPI_NAMES)})."
        )
    return {
        name: values[:, index].tolist()
        for index, name in enumerate(KPI_NAMES)
    }


def _validate_sampling_rate(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError("sampling_rate_hz must be a finite positive number.")
    return float(value)
