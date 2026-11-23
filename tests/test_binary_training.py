from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from cl5g.common.exceptions import SchemaValidationError
from cl5g.data.telecomts import KPI_NAMES
from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor
from cl5g.features.extraction import build_feature_configuration
from cl5g.pipeline.train import _load_partition


def _write_partition(directory: Path, *, invalid_kpis: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sequences = []
    metadata = []
    for index, label in enumerate(("Normal", "Jamming")):
        kpis = {
            name: (
                ["TCP" if index == 0 else "UDP"] * 128
                if name in {"UL_Protocol", "DL_Protocol"}
                else [float(step + index) for step in range(128)]
            )
            for name in KPI_NAMES
        }
        if invalid_kpis and index == 1:
            kpis.pop("RSRP")
        sample_id = f"sample-{index}"
        sequences.append({"sample_id": sample_id, "kpis": kpis})
        metadata.append(
            {
                "sample_id": sample_id,
                "source_file_id": f"source-{index}",
                "label": label,
                "source_anomaly_type": label,
            }
        )
    for name, records in (
        ("train_sequences.jsonl.gz", sequences),
        ("train_metadata.jsonl.gz", metadata),
    ):
        with gzip.open(directory / name, "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")


def test_training_loader_preserves_labels_ids_and_ordered_shared_features(
    tmp_path: Path,
) -> None:
    _write_partition(tmp_path)
    feature_configuration = build_feature_configuration("base_plus_temporal")

    partition = _load_partition(
        tmp_path,
        "train",
        feature_configuration,
        TelecomTSPreprocessor(),
        batch_size=1,
    )

    assert partition.features.shape == (2, 250)
    assert list(partition.features.columns) == feature_configuration["feature_order"]
    assert partition.binary_labels.tolist() == [0, 1]
    assert partition.labels == ("Normal", "Jamming")
    assert partition.sample_ids == ("sample-0", "sample-1")
    assert partition.source_ids == ("source-0", "source-1")
    assert np.isfinite(partition.features.to_numpy()).all()


def test_training_loader_rejects_malformed_kpi_channels(tmp_path: Path) -> None:
    _write_partition(tmp_path, invalid_kpis=True)

    with pytest.raises(SchemaValidationError, match="Invalid KPI channels"):
        _load_partition(
            tmp_path,
            "train",
            build_feature_configuration("base_plus_temporal"),
            TelecomTSPreprocessor(),
        )
