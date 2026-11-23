"""Export a reproducibly sampled TelecomTS history to dashboard-ready CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from itertools import zip_longest
from pathlib import Path
from typing import Any

from cl5g.data.telecomts import (
    ALLOWED_PROTOCOL_VALUES,
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TAXONOMY_LABELS,
    TIME_STEPS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REVISION = DEFAULT_REVISION
SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42
MAX_OUTPUT_BYTES = 190 * 1024 * 1024
MIB = 1024 * 1024
CSV_COLUMNS = ("timestamp", "entity_id", "layer", "anomaly_type", *KPI_NAMES)
_MAX_INVALID_EXAMPLES = 100


class HistoricalCsvError(Exception):
    """A clear, actionable error while reading or exporting TelecomTS data."""


def generate_historical_csv(
    sequences_path: Path,
    metadata_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> dict[str, Any]:
    """Stream, validate, stratify-sample, and export ingested TelecomTS samples."""
    for path, label in (
        (sequences_path, "Processed sequences"),
        (metadata_path, "Processed metadata"),
    ):
        if not path.is_file():
            raise HistoricalCsvError(
                f"{label} file not found: {path}. Run "
                f"'python -m cl5g.data.telecomts --offline' from the project root "
                "to rebuild processed data from already-downloaded raw files. "
                "Offline ingestion will not download TelecomTS again."
            )
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be positive.")
    if sequences_path.resolve().parent != metadata_path.resolve().parent:
        raise HistoricalCsvError(
            "Sequences and metadata must come from the same revision directory."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_path.parent).free < MIB:
        raise HistoricalCsvError(
            f"Less than 1 MiB is available at {output_path.parent}; "
            "free disk space before generating the CSV."
        )

    reservoirs, scan_report = _sample_processed_records(
        sequences_path,
        metadata_path,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    available_counts = scan_report["valid_samples_per_class"]
    if scan_report["aligned_records"] == 0:
        raise HistoricalCsvError(
            "The processed JSONL files contain no aligned records to export."
        )
    if sum(len(records) for records in reservoirs.values()) == 0:
        raise HistoricalCsvError(
            "No valid samples were available for the configured taxonomy."
        )

    rng = random.Random(seed)
    for records in reservoirs.values():
        rng.shuffle(records)

    sample_limit = samples_per_class
    while True:
        selected = {
            label: reservoirs[label][:sample_limit]
            for label in TAXONOMY_LABELS
        }
        selected_counts = {
            label: len(selected[label])
            for label in TAXONOMY_LABELS
        }
        selected_total = sum(selected_counts.values())
        if selected_total == 0:
            raise HistoricalCsvError(
                "Sampling produced no records; check the processed input files."
            )

        temp_path = _temporary_path(output_path)
        try:
            missing_values = _write_csv(temp_path, selected)
            _validate_written_csv(
                temp_path,
                selected_counts=selected_counts,
                expected_rows=selected_total * TIME_STEPS,
            )
            size_bytes = temp_path.stat().st_size
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            if getattr(exc, "errno", None) == 28:
                raise HistoricalCsvError(
                    f"Insufficient disk space while writing {output_path}; "
                    "free space and rerun the generator."
                ) from exc
            raise HistoricalCsvError(
                f"Could not write or validate CSV at {output_path}: {exc}"
            ) from exc
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        if size_bytes <= max_output_bytes:
            os.replace(temp_path, output_path)
            break
        temp_path.unlink(missing_ok=True)
        if sample_limit == 1:
            raise HistoricalCsvError(
                f"The CSV is still {size_bytes / MIB:.1f} MiB at one sample per "
                f"class, above the {max_output_bytes / MIB:.1f} MiB limit."
            )
        reduced_limit = max(
            1,
            min(sample_limit - 1, int(sample_limit * max_output_bytes / size_bytes * 0.9)),
        )
        print(
            f"CSV was {size_bytes / MIB:.1f} MiB, over the "
            f"{max_output_bytes / MIB:.1f} MiB limit; retrying with at most "
            f"{reduced_limit} samples per class.",
            file=sys.stderr,
        )
        sample_limit = reduced_limit

    report = {
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": REVISION,
        "input_files": {
            "sequences": str(sequences_path.resolve()),
            "metadata": str(metadata_path.resolve()),
        },
        "output_file": str(output_path.resolve()),
        "requested_samples_per_class": samples_per_class,
        "final_samples_per_class_limit": sample_limit,
        "random_seed": seed,
        "sampling_method": "reservoir sampling independently by canonical anomaly class",
        "layer_value": "ran_du",
        "layer_note": (
            "Assigned for dashboard compatibility; TelecomTS does not provide "
            "an original cross-layer annotation for this field."
        ),
        "total_source_records_aligned": scan_report["aligned_records"],
        "total_valid_samples": sum(available_counts.values()),
        "total_selected_samples": selected_total,
        "samples_per_class_available": available_counts,
        "samples_per_class_selected": selected_counts,
        "total_output_rows": selected_total * TIME_STEPS,
        "invalid_samples_skipped": scan_report["invalid_samples"],
        "invalid_sample_examples": scan_report["invalid_sample_examples"],
        "invalid_example_count_truncated": scan_report["invalid_example_count_truncated"],
        "missing_values": missing_values,
        "output_size_bytes": size_bytes,
        "output_size_mib": round(size_bytes / MIB, 3),
        "maximum_output_size_bytes": max_output_bytes,
        "csv_columns": list(CSV_COLUMNS),
        "kpi_order": list(KPI_NAMES),
    }
    _write_json_atomic(report_path, report)
    return report


def _sample_processed_records(
    sequences_path: Path,
    metadata_path: Path,
    *,
    samples_per_class: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    reservoirs: dict[str, list[dict[str, Any]]] = {
        label: [] for label in TAXONOMY_LABELS
    }
    class_seen: Counter[str] = Counter()
    rng = random.Random(seed)
    invalid_examples: list[dict[str, Any]] = []
    invalid_count = 0
    aligned_count = 0
    seen_sample_ids: set[str] = set()

    try:
        sequence_stream = gzip.open(sequences_path, "rt", encoding="utf-8")
        metadata_stream = gzip.open(metadata_path, "rt", encoding="utf-8")
    except (OSError, EOFError) as exc:
        raise HistoricalCsvError(
            f"Could not open processed gzip JSONL files: {exc}"
        ) from exc

    with sequence_stream, metadata_stream:
        try:
            pairs = zip_longest(sequence_stream, metadata_stream)
            for line_number, (sequence_line, metadata_line) in enumerate(pairs, start=1):
                aligned_count += 1
                errors: list[str] = []
                sequence: Any = None
                metadata: Any = None
                if sequence_line is None or metadata_line is None:
                    errors.append("Sequences and metadata JSONL record counts differ.")
                else:
                    try:
                        sequence = json.loads(sequence_line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"Malformed sequence JSON: {exc.msg}.")
                    try:
                        metadata = json.loads(metadata_line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"Malformed metadata JSON: {exc.msg}.")

                if not errors:
                    errors.extend(_validate_record(sequence, metadata, seen_sample_ids))
                if errors:
                    invalid_count += 1
                    if len(invalid_examples) < _MAX_INVALID_EXAMPLES:
                        invalid_examples.append(
                            {
                                "line_number": line_number,
                                "sample_id": _record_sample_id(sequence, metadata),
                                "errors": errors,
                            }
                        )
                    print(
                        f"Skipping invalid processed sample at JSONL line "
                        f"{line_number}: {'; '.join(errors)}",
                        file=sys.stderr,
                    )
                    continue

                sample_id = sequence["sample_id"]
                seen_sample_ids.add(sample_id)
                label = metadata["label"]
                class_seen[label] += 1
                sample = {
                    "sample_id": sample_id,
                    "start_time": metadata["start_time"],
                    "sampling_rate": metadata["sampling_rate"],
                    "label": label,
                    "kpis": sequence["kpis"],
                }
                reservoir = reservoirs[label]
                if len(reservoir) < samples_per_class:
                    reservoir.append(sample)
                else:
                    replacement_index = rng.randrange(class_seen[label])
                    if replacement_index < samples_per_class:
                        reservoir[replacement_index] = sample
        except (OSError, EOFError, gzip.BadGzipFile, UnicodeError) as exc:
            raise HistoricalCsvError(
                f"Could not read compressed JSONL data: {exc}. Check that both "
                "processed gzip files are complete and from the same revision."
            ) from exc

    return reservoirs, {
        "aligned_records": aligned_count,
        "invalid_samples": invalid_count,
        "invalid_sample_examples": invalid_examples,
        "invalid_example_count_truncated": max(0, invalid_count - len(invalid_examples)),
        "valid_samples_per_class": {
            label: int(class_seen.get(label, 0))
            for label in TAXONOMY_LABELS
        },
    }


def _validate_record(
    sequence: Any,
    metadata: Any,
    seen_sample_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(sequence, dict):
        return ["Sequence record must be a JSON object."]
    if not isinstance(metadata, dict):
        return ["Metadata record must be a JSON object."]

    sequence_id = sequence.get("sample_id")
    metadata_id = metadata.get("sample_id")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        errors.append("Sequence sample_id must be a non-empty string.")
    if not isinstance(metadata_id, str) or not metadata_id.strip():
        errors.append("Metadata sample_id must be a non-empty string.")
    if isinstance(sequence_id, str) and isinstance(metadata_id, str):
        if sequence_id != metadata_id:
            errors.append("Sequence and metadata sample_id values do not match.")
        if sequence_id in seen_sample_ids:
            errors.append(f"Duplicate sample_id {sequence_id!r}.")

    kpis = sequence.get("kpis")
    if not isinstance(kpis, dict):
        errors.append("Sequence kpis field must be an object.")
        kpis = {}
    missing_kpis = [name for name in KPI_NAMES if name not in kpis]
    extra_kpis = [str(name) for name in kpis if name not in KPI_NAMES]
    if missing_kpis:
        errors.append("Missing KPI channel(s): " + ", ".join(missing_kpis) + ".")
    if extra_kpis:
        errors.append("Unexpected KPI channel(s): " + ", ".join(extra_kpis) + ".")
    for name in KPI_NAMES:
        values = kpis.get(name)
        if not isinstance(values, list) or len(values) != TIME_STEPS:
            actual_length = len(values) if isinstance(values, list) else "not a list"
            errors.append(
                f"KPI {name!r} must contain exactly {TIME_STEPS} values "
                f"(received {actual_length})."
            )
            continue
        if name in NUMERIC_KPIS:
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                errors.append(f"Numeric KPI {name!r} has a missing or invalid value.")
        elif any(
            not isinstance(value, str) or value not in ALLOWED_PROTOCOL_VALUES
            for value in values
        ):
            errors.append(
                f"Categorical KPI {name!r} must use empty, None, TCP, or UDP strings."
            )

    start_time = metadata.get("start_time")
    try:
        if not isinstance(start_time, str):
            raise ValueError("must be a timestamp string")
        datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError as exc:
        errors.append(f"Invalid start_time: {exc}.")

    sampling_rate = metadata.get("sampling_rate")
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, (int, float))
        or not math.isfinite(float(sampling_rate))
        or sampling_rate <= 0
    ):
        errors.append("sampling_rate must be a finite positive number.")

    label = metadata.get("label")
    if not isinstance(label, str) or label not in TAXONOMY_LABELS:
        errors.append("Metadata label is missing or outside the TelecomTS taxonomy.")
    return errors


def _record_sample_id(sequence: Any, metadata: Any) -> str | None:
    if isinstance(sequence, dict) and isinstance(sequence.get("sample_id"), str):
        return sequence["sample_id"]
    if isinstance(metadata, dict) and isinstance(metadata.get("sample_id"), str):
        return metadata["sample_id"]
    return None


def _write_csv(
    path: Path,
    selected: dict[str, list[dict[str, Any]]],
) -> int:
    missing_values = 0
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="raise")
            writer.writeheader()
            for label in TAXONOMY_LABELS:
                for sample in selected[label]:
                    start_time = datetime.fromisoformat(
                        sample["start_time"].replace("Z", "+00:00")
                    )
                    sampling_rate = float(sample["sampling_rate"])
                    for timestep in range(TIME_STEPS):
                        timestamp = start_time + timedelta(
                            seconds=timestep / sampling_rate
                        )
                        row: dict[str, Any] = {
                            "timestamp": timestamp.isoformat(
                                sep=" ",
                                timespec="microseconds",
                            ),
                            "entity_id": sample["sample_id"],
                            "layer": "ran_du",
                            "anomaly_type": label,
                        }
                        for kpi in KPI_NAMES:
                            value = sample["kpis"][kpi][timestep]
                            if value is None:
                                missing_values += 1
                            row[kpi] = value
                        writer.writerow(row)
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise
        raise HistoricalCsvError(f"Could not create CSV file {path}: {exc}") from exc
    return missing_values


def _validate_written_csv(
    path: Path,
    *,
    selected_counts: dict[str, int],
    expected_rows: int,
) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise HistoricalCsvError(
                "Generated CSV columns do not match the canonical dashboard schema."
            )
        row_count = 0
        rows_by_sample: Counter[str] = Counter()
        labels_by_sample: dict[str, str] = {}
        for row in reader:
            row_count += 1
            sample_id = row["entity_id"]
            if not sample_id:
                raise HistoricalCsvError("Generated CSV contains an empty entity_id.")
            if row["layer"] != "ran_du":
                raise HistoricalCsvError("Generated CSV contains an invalid layer value.")
            if row["anomaly_type"] not in TAXONOMY_LABELS:
                raise HistoricalCsvError("Generated CSV contains an invalid anomaly label.")
            previous_label = labels_by_sample.setdefault(sample_id, row["anomaly_type"])
            if previous_label != row["anomaly_type"]:
                raise HistoricalCsvError(
                    f"Generated CSV sample {sample_id!r} has inconsistent labels."
                )
            rows_by_sample[sample_id] += 1
            for kpi in NUMERIC_KPIS:
                if row[kpi] == "":
                    raise HistoricalCsvError(
                        f"Generated CSV has a missing numeric value in {kpi}."
                    )
                try:
                    value = float(row[kpi])
                except ValueError as exc:
                    raise HistoricalCsvError(
                        f"Generated CSV has a non-numeric value in {kpi}."
                    ) from exc
                if not math.isfinite(value):
                    raise HistoricalCsvError(
                        f"Generated CSV has a non-finite value in {kpi}."
                    )
            for kpi in ("UL_Protocol", "DL_Protocol"):
                if row[kpi] not in ALLOWED_PROTOCOL_VALUES:
                    raise HistoricalCsvError(
                        f"Generated CSV has an invalid protocol value in {kpi}."
                    )
            try:
                datetime.fromisoformat(row["timestamp"])
            except ValueError as exc:
                raise HistoricalCsvError(
                    "Generated CSV contains an invalid timestamp."
                ) from exc

    if row_count != expected_rows:
        raise HistoricalCsvError(
            f"Generated CSV has {row_count} rows; expected {expected_rows}."
        )
    expected_samples = sum(selected_counts.values())
    if len(rows_by_sample) != expected_samples:
        raise HistoricalCsvError(
            f"Generated CSV contains {len(rows_by_sample)} sample IDs; "
            f"expected {expected_samples}."
        )
    incorrect = {
        sample_id: count
        for sample_id, count in rows_by_sample.items()
        if count != TIME_STEPS
    }
    if incorrect:
        sample_id, count = next(iter(incorrect.items()))
        raise HistoricalCsvError(
            f"Sample {sample_id!r} has {count} CSV rows; expected {TIME_STEPS}."
        )
    observed_class_counts = Counter(labels_by_sample.values())
    expected_class_counts = {
        label: count
        for label, count in selected_counts.items()
        if count > 0
    }
    if dict(observed_class_counts) != expected_class_counts:
        raise HistoricalCsvError(
            "Generated CSV class counts do not match the selected sample counts."
        )


def _temporary_path(output_path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temp_path, path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        if getattr(exc, "errno", None) == 28:
            raise HistoricalCsvError(
                f"Insufficient disk space while writing report {path}."
            ) from exc
        raise HistoricalCsvError(f"Could not write report {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export actual ingested TelecomTS sequences for historical dashboard exploration."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=SAMPLES_PER_CLASS,
        help=f"Maximum records per category (default: {SAMPLES_PER_CLASS}).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--sequences",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "telecomts"
        / REVISION
        / "sequences.jsonl.gz",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "telecomts"
        / REVISION
        / "metadata.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "telecomts_historical_exploration.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "reports"
        / "telecomts"
        / "historical_csv_report.json",
    )
    args = parser.parse_args(argv)
    try:
        report = generate_historical_csv(
            args.sequences,
            args.metadata,
            args.output,
            args.report,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    except (HistoricalCsvError, OSError, ValueError) as exc:
        print(f"Historical CSV generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generated: {report['output_file']}")
    print(f"Dataset revision: {report['dataset_revision']}")
    print(
        f"Selected samples: {report['total_selected_samples']} "
        f"({report['total_output_rows']} rows)"
    )
    for label, count in report["samples_per_class_selected"].items():
        print(f"  {label}: {count}")
    print(
        f"Invalid samples skipped: {report['invalid_samples_skipped']} "
        f"(details in {args.report})"
    )
    print(
        f"CSV size: {report['output_size_bytes']:,} bytes "
        f"({report['output_size_mib']:.3f} MiB)"
    )
    print(f"Validation report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
