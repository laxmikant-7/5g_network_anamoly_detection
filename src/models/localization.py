"""Localization model interface.

Future training code will call ``fit_localizer``. Future inference code will
call ``predict_layer``. Saved estimators are expected to use joblib, which
this module does not load.
"""

from __future__ import annotations

import pandas as pd


def fit_localizer(features: pd.DataFrame, layers: pd.Series) -> None:
    """Fit a model that predicts the anomalous layer.

    Raises:
        TypeError: If an argument has the wrong type.
        NotImplementedError: Localization training is not implemented.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if not isinstance(layers, pd.Series):
        raise TypeError("layers must be a pandas Series.")
    raise NotImplementedError("Localization training is not implemented.")


def predict_layer(model: object, features: pd.DataFrame) -> pd.Series:
    """Predict a layer label for each feature row.

    Raises:
        TypeError: If an argument has the wrong type.
        NotImplementedError: Localization inference is not implemented.
    """
    if model is None:
        raise TypeError("model is required.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    raise NotImplementedError("Localization inference is not implemented.")
