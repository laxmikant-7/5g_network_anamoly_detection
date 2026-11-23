"""Reserved metric functions.

Planned measures are layer-localization accuracy and diagnosis macro-F1.
Neither function calculates a score in this scaffold.
"""

from __future__ import annotations

import pandas as pd


def score_localization(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Score layer predictions.

    Raises:
        TypeError: If an argument has the wrong type.
        NotImplementedError: Localization metrics are not implemented.
    """
    _require_series(y_true, "y_true")
    _require_series(y_pred, "y_pred")
    raise NotImplementedError("Localization metrics are not implemented.")


def score_diagnosis(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Score anomaly-type predictions.

    Raises:
        TypeError: If an argument has the wrong type.
        NotImplementedError: Diagnosis metrics are not implemented.
    """
    _require_series(y_true, "y_true")
    _require_series(y_pred, "y_pred")
    raise NotImplementedError("Diagnosis metrics are not implemented.")


def _require_series(value: object, name: str) -> None:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
