"""Reproducible TelecomTS download, validation, and sequence ingestion."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

from cl5g.common.exceptions import ConfigurationError, ProjectError
from cl5g.common.logging_config import configure_logging, get_logger

DATASET_ID = "AliMaatouk/TelecomTS"
DEFAULT_REVISION = "1b3a88a440f4922edfcd1e230e42c003c0e13bdf"
TIME_STEPS = 128
NUMERIC_KPIS = (
    "RSRP",
    "DL_BLER",
    "DL_MCS",
    "UL_BLER",
    "UL_MCS",
    "UL_NPRB",
    "UL_SNR",
    "TX_Bytes",
    "RX_Bytes",
    "Estimated_UL_Buffer",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "UL_NumberOfPackets",
    "DL_NumberOfPackets",
)
CATEGORICAL_KPIS = ("UL_Protocol", "DL_Protocol")
KPI_NAMES = (
    "RSRP",
    "DL_BLER",
    "DL_MCS",
    "UL_BLER",
    "UL_MCS",
    "UL_NPRB",
    "UL_SNR",
    "TX_Bytes",
    "RX_Bytes",
    "Estimated_UL_Buffer",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "UL_Protocol",
    "UL_NumberOfPackets",
    "DL_Protocol",
    "DL_NumberOfPackets",
)
TAXONOMY_LABELS = (
    "Normal",
    "Jamming",
    "Co-Channel Interference (Mild)",
    "Buffer Overflow (Gradual Buildup)",
    "Co-Channel Interference (Severe)",
    "Antenna Failure",
    "Faulty RF Filters (Temporal)",
    "High Network Congestion (Gradual Buildup)",
    "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)",
    "Resource Allocation Bugs",
    "High Network Congestion (Sudden Spike)",
)
LABEL_MAPPING = {
    label.casefold(): label
    for label in TAXONOMY_LABELS[1:]
}
LABEL_MAPPING.update({"": "Normal", "normal": "Normal", "none": "Normal"})

# TelecomTS uses an empty protocol string to represent no protocol at that time
# step. It is retained as a categorical value; JSON null and non-string values
# remain validation errors.
ALLOWED_PROTOCOL_VALUES = frozenset({"", "None", "TCP", "UDP"})
_MAX_ERROR_LOG_MESSAGES = 20
_LOGGER = get_logger("data.telecomts")


class TelecomTSError(ProjectError):
    """A TelecomTS source or ingestion operation failed."""


@dataclass(frozen=True)
class SourceFile:
    """One immutable source-file entry from the Hugging Face repository tree."""

    path: str
    size: int
    oid: str
    lfs_sha256: str | None = None


def discover_source_files(revision: str = DEFAULT_REVISION) -> list[SourceFile]:
    """Discover all processed TelecomTS JSONL sources at an immutable revision."""
    _validate_revision(revision)
    url = (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{revision}?recursive=true&expand=false"
    )
    payload = _get_json(url, "TelecomTS repository tree")
    if not isinstance(payload, list):
        raise TelecomTSError("Hugging Face returned an invalid repository tree.")
    sources: list[SourceFile] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if item.get("type") != "file" or not isinstance(path, str):
            continue
        if not path.endswith("/processed/chunked.jsonl"):
            continue
        _validate_source_path(path)
        size = item.get("size")
        oid = item.get("oid")
        lfs = item.get("lfs")
        lfs_oid = lfs.get("oid") if isinstance(lfs, dict) else None
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(oid, str)
        ):
            raise TelecomTSError(f"Invalid source metadata for '{path}'.")
        sources.append(
            SourceFile(
                path=path,
                size=size,
                oid=oid,
                lfs_sha256=lfs_oid if isinstance(lfs_oid, str) else None,
            )
        )
    sources.sort(key=lambda source: source.path)
    if not sources:
        raise TelecomTSError("No processed TelecomTS chunked.jsonl files were found.")
    return sources


def ingest_telecomts(
    project_root: Path | None = None,
    *,
    revision: str = DEFAULT_REVISION,
    offline: bool = False,
) -> dict[str, Any]:
    """Download (unless offline) and ingest all pinned TelecomTS source files."""
    _validate_revision(revision)
    root = Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    raw_directory = root / "data" / "raw" / "telecomts" / revision
    processed_directory = root / "data" / "processed" / "telecomts" / revision
    report_directory = root / "data" / "reports" / "telecomts" / revision

    if offline:
        source_manifest = _read_source_manifest(raw_directory, revision)
        sources = [
            SourceFile(
                path=item["path"],
                size=int(item["size"]),
                oid=item["oid"],
                lfs_sha256=item.get("lfs_sha256"),
            )
            for item in source_manifest["files"]
        ]
    else:
        sources = discover_source_files(revision)
        source_manifest = _download_sources(raw_directory, revision, sources)

    local_sources = [
        (source, raw_directory / (source.path + ".gz"))
        for source in sources
    ]
    summary = process_source_files(
        local_sources,
        revision=revision,
        processed_directory=processed_directory,
        report_directory=report_directory,
    )
    summary["source_manifest"] = _relative(root, raw_directory / "source_manifest.json")
    summary["sequence_file"] = _relative(root, processed_directory / "sequences.jsonl.gz")
    summary["metadata_file"] = _relative(root, processed_directory / "metadata.jsonl.gz")
    summary["validation_report"] = _relative(root, report_directory / "validation.json")
    summary["invalid_records_report"] = _relative(
        root, report_directory / "invalid_records.jsonl.gz"
    )
    _write_json_atomic(report_directory / "validation.json", summary)
    _LOGGER.info(
        "TelecomTS ingestion complete: %s loaded, %s invalid, %s source files.",
        summary["loaded_samples"],
        summary["invalid_samples"],
        summary["source_file_count"],
    )
    return summary


def process_source_files(
    source_files: Sequence[tuple[SourceFile, Path]],
    *,
    revision: str,
    processed_directory: Path,
    report_directory: Path,
) -> dict[str, Any]:
    """Validate local compressed JSONL sources and atomically write outputs."""
    if not source_files:
        raise TelecomTSError("No local TelecomTS source files were provided.")
    processed_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    sequence_tmp = processed_directory / "sequences.jsonl.gz.tmp"
    metadata_tmp = processed_directory / "metadata.jsonl.gz.tmp"
    errors_tmp = report_directory / "invalid_records.jsonl.gz.tmp"
    counters: Counter[str] = Counter()
    labels_seen: Counter[str] = Counter()
    raw_labels_seen: Counter[str] = Counter()
    valid_labels: Counter[str] = Counter()
    source_line_counts: dict[str, int] = {}
    protocol_values: dict[str, set[str]] = {name: set() for name in CATEGORICAL_KPIS}
    seen_sample_ids: set[str] = set()
    logged_errors = 0
    try:
        with (
            _gzip_writer(sequence_tmp) as sequences_out,
            _gzip_writer(metadata_tmp) as metadata_out,
            _gzip_writer(errors_tmp) as errors_out,
        ):
            for source, source_path in source_files:
                if not source_path.is_file():
                    raise TelecomTSError(
                        f"Downloaded source file is missing: '{source_path}'."
                    )
                _LOGGER.info("Validating source file %s.", source.path)
                lines_in_file = 0
                with gzip.open(source_path, "rb") as source_in:
                    for line_number, raw_bytes in enumerate(source_in, start=1):
                        lines_in_file = line_number
                        counters["source_lines"] += 1
                        errors: list[str] = []
                        try:
                            raw_line = raw_bytes.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            record = None
                            errors.append(f"Malformed UTF-8: {exc.reason}.")
                        else:
                            try:
                                record = json.loads(raw_line)
                            except json.JSONDecodeError as exc:
                                record = None
                                errors.append(
                                    f"Malformed JSON: {exc.msg} (column {exc.colno})."
                                )
                        if record is None and not errors:
                            errors.append("Record must be a JSON object; null is not a sample.")
                        if not errors and not isinstance(record, dict):
                            errors.append("Record must be a JSON object.")

                        if isinstance(record, dict):
                            counters["parsed_records"] += 1
                            sample_id, has_source_id = _sample_id(
                                record, revision, source.path, line_number
                            )
                            if sample_id in seen_sample_ids:
                                errors.append(f"Duplicate sample ID '{sample_id}'.")
                            else:
                                seen_sample_ids.add(sample_id)
                            sample_errors, normalized = validate_record(record)
                            errors.extend(sample_errors)
                            if normalized is not None:
                                labels_seen[normalized["anomaly_type"]] += 1
                                raw_labels_seen[normalized["source_anomaly_type"]] += 1
                                _count_annotations(normalized, counters)
                                _collect_protocol_values(normalized["kpis"], protocol_values)

                            if not errors and normalized is not None:
                                valid_labels[normalized["anomaly_type"]] += 1
                                counters["loaded_samples"] += 1
                                sequence_record = {
                                    "sample_id": sample_id,
                                    "kpis": normalized["kpis"],
                                }
                                metadata_record = {
                                    "sample_id": sample_id,
                                    "source_file_id": source.path,
                                    "source_line_number": line_number,
                                    "source_sample_id_present": has_source_id,
                                    "start_time": record.get("start_time"),
                                    "end_time": record.get("end_time"),
                                    "sampling_rate": record.get("sampling_rate"),
                                    "label": normalized["anomaly_type"],
                                    "binary_label": normalized["binary_label"],
                                    "source_anomaly_type": normalized["source_anomaly_type"],
                                    "anomalies": normalized["annotations"],
                                    "labels": normalized["labels"],
                                }
                                _write_json_line(sequences_out, sequence_record)
                                _write_json_line(metadata_out, metadata_record)
                            else:
                                counters["invalid_samples"] += 1
                                error_entry = {
                                    "source_file_id": source.path,
                                    "source_line_number": line_number,
                                    "sample_id": sample_id,
                                    "errors": errors,
                                }
                                _write_json_line(errors_out, error_entry)
                                if logged_errors < _MAX_ERROR_LOG_MESSAGES:
                                    _LOGGER.error(
                                        "Invalid TelecomTS record %s:%s: %s",
                                        source.path,
                                        line_number,
                                        "; ".join(errors),
                                    )
                                    logged_errors += 1
                        else:
                            counters["invalid_samples"] += 1
                            _write_json_line(
                                errors_out,
                                {
                                    "source_file_id": source.path,
                                    "source_line_number": line_number,
                                    "sample_id": None,
                                    "errors": errors,
                                },
                            )
                            if logged_errors < _MAX_ERROR_LOG_MESSAGES:
                                _LOGGER.error(
                                    "Invalid TelecomTS record %s:%s: %s",
                                    source.path,
                                    line_number,
                                    "; ".join(errors),
                                )
                                logged_errors += 1
                source_line_counts[source.path] = lines_in_file

        _replace_file(sequence_tmp, processed_directory / "sequences.jsonl.gz")
        _replace_file(metadata_tmp, processed_directory / "metadata.jsonl.gz")
        _replace_file(errors_tmp, report_directory / "invalid_records.jsonl.gz")
    except Exception:
        for path in (sequence_tmp, metadata_tmp, errors_tmp):
            path.unlink(missing_ok=True)
        raise

    if counters["invalid_samples"] > logged_errors:
        _LOGGER.error(
            "%s additional invalid records are listed in the validation error report.",
            counters["invalid_samples"] - logged_errors,
        )

    distribution = {label: int(valid_labels.get(label, 0)) for label in TAXONOMY_LABELS}
    observed_distribution = {
        label: int(labels_seen.get(label, 0)) for label in TAXONOMY_LABELS
    }
    return {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "format": "JSON Lines; one nested sample per line",
        "source_file_count": len(source_files),
        "source_files": [source.path for source, _ in source_files],
        "source_line_count": int(counters["source_lines"]),
        "parsed_record_count": int(counters["parsed_records"]),
        "loaded_samples": int(counters["loaded_samples"]),
        "normal_samples": int(valid_labels.get("Normal", 0)),
        "anomalous_samples": int(
            sum(value for label, value in valid_labels.items() if label != "Normal")
        ),
        "invalid_samples": int(counters["invalid_samples"]),
        "anomaly_category_distribution": distribution,
        "parsed_label_distribution_including_invalid_records": observed_distribution,
        "source_anomaly_type_distribution": dict(sorted(raw_labels_seen.items())),
        "samples_with_temporal_annotations": int(counters["temporal_annotations"]),
        "samples_with_affected_kpi_annotations": int(counters["affected_kpi_annotations"]),
        "samples_missing_temporal_annotations": int(counters["missing_temporal_annotations"]),
        "samples_missing_affected_kpi_annotations": int(
            counters["missing_affected_kpi_annotations"]
        ),
        "categorical_values": {
            name: sorted(values) for name, values in protocol_values.items()
        },
        "source_line_counts": source_line_counts,
        "validation_error_count": int(counters["invalid_samples"]),
    }


def validate_record(record: Mapping[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    """Validate and normalize one original TelecomTS sample without mutating it."""
    errors: list[str] = []
    kpis = record.get("KPIs")
    if not isinstance(kpis, dict):
        errors.append("KPIs must be an object.")
        kpis = {}
    expected_names = set(KPI_NAMES)
    actual_names = set(kpis)
    missing_kpis = sorted(expected_names - actual_names)
    extra_kpis = sorted(str(name) for name in actual_names - expected_names)
    if missing_kpis:
        errors.append("Missing KPI(s): " + ", ".join(missing_kpis) + ".")
    if extra_kpis:
        errors.append("Unexpected KPI(s): " + ", ".join(extra_kpis) + ".")

    normalized_kpis: dict[str, list[Any]] = {}
    for name in KPI_NAMES:
        values = kpis.get(name)
        if not isinstance(values, list):
            errors.append(f"KPI '{name}' must be a list.")
            continue
        if len(values) != TIME_STEPS:
            errors.append(
                f"KPI '{name}' must contain {TIME_STEPS} values; received {len(values)}."
            )
            continue
        if name in NUMERIC_KPIS:
            for index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"KPI '{name}' at time step {index} is not numeric.")
                    break
                if not math.isfinite(float(value)):
                    errors.append(f"KPI '{name}' at time step {index} is not finite.")
                    break
        else:
            for index, value in enumerate(values):
                if not isinstance(value, str):
                    errors.append(
                        f"KPI '{name}' at time step {index} is missing or not a string."
                    )
                    break
                if value not in ALLOWED_PROTOCOL_VALUES:
                    errors.append(
                        f"KPI '{name}' has unsupported categorical value {value!r} "
                        f"at time step {index}."
                    )
                    break
        normalized_kpis[name] = values

    for field in ("start_time", "end_time"):
        value = record.get(field)
        if not isinstance(value, str) or not _valid_timestamp(value):
            errors.append(f"'{field}' must be a valid timestamp string.")
    rate = record.get("sampling_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or rate <= 0
    ):
        errors.append("'sampling_rate' must be a positive number.")

    anomalies = record.get("anomalies")
    if not isinstance(anomalies, dict):
        errors.append("'anomalies' must be an object.")
        anomalies = {}
    exists = anomalies.get("exists")
    if not isinstance(exists, bool):
        errors.append("'anomalies.exists' must be a boolean.")
    source_type = anomalies.get("type")
    if not isinstance(source_type, str):
        errors.append("'anomalies.type' must be a string.")
        source_type = ""
    canonical_type = LABEL_MAPPING.get(source_type.strip().casefold())
    if isinstance(exists, bool):
        if exists and canonical_type in (None, "Normal"):
            errors.append(
                f"Unknown anomalous source label {source_type!r}; no taxonomy mapping exists."
            )
        if not exists:
            canonical_type = "Normal"

    labels = record.get("labels")
    if not isinstance(labels, dict):
        errors.append("'labels' must be an object.")
        labels = {}
    presence = labels.get("anomaly_present")
    if not isinstance(presence, str) or presence.strip().casefold() not in {
        "yes",
        "no",
        "true",
        "false",
    }:
        errors.append("'labels.anomaly_present' must be Yes/No or True/False.")
    elif isinstance(exists, bool):
        label_is_anomaly = presence.strip().casefold() in {"yes", "true"}
        if label_is_anomaly != exists:
            errors.append("'labels.anomaly_present' disagrees with 'anomalies.exists'.")

    duration = anomalies.get("anomaly_duration")
    affected = anomalies.get("affected_kpis")
    if exists is True:
        if not isinstance(duration, dict):
            errors.append("Anomalous sample is missing 'anomalies.anomaly_duration'.")
        else:
            start = duration.get("start")
            end = duration.get("end")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end >= TIME_STEPS
            ):
                errors.append(
                    "Anomalous sample has an invalid inclusive anomaly duration."
                )
        if not isinstance(affected, list) or not any(
            isinstance(name, str) and name.strip() for name in affected
        ):
            errors.append("Anomalous sample is missing affected KPI annotations.")
        elif any(
            not isinstance(name, str)
            or (name.strip() and name not in expected_names)
            for name in affected
        ):
            errors.append("Anomalous sample has an unknown affected KPI annotation.")

    normalized = None
    if isinstance(exists, bool) and canonical_type is not None:
        normalized = {
            "kpis": normalized_kpis,
            "anomaly_type": canonical_type,
            "binary_label": exists,
            "source_anomaly_type": source_type,
            "annotations": {
                key: anomalies.get(key)
                for key in ("exists", "type", "anomaly_duration", "affected_kpis")
            },
            "labels": dict(labels),
        }
    return errors, normalized


def _download_sources(
    raw_directory: Path,
    revision: str,
    sources: Sequence[SourceFile],
) -> dict[str, Any]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for source in sources:
        target = raw_directory / (source.path + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and _verify_source_archive(target, source):
            _LOGGER.info("Using cached source file %s.", source.path)
            raw_sha256 = _sha256_gzip_contents(target)
        else:
            raw_sha256 = _download_source(target, source, revision)
        files.append(
            {
                "path": source.path,
                "size": source.size,
                "oid": source.oid,
                "lfs_sha256": source.lfs_sha256,
                "downloaded_content_sha256": raw_sha256,
                "local_path": str(target.relative_to(raw_directory)).replace("\\", "/"),
            }
        )
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "source_file_count": len(files),
        "source_bytes_uncompressed": sum(item["size"] for item in files),
        "storage_format": "gzip-compressed original chunked.jsonl files",
        "files": files,
    }
    _write_json_atomic(raw_directory / "source_manifest.json", manifest)
    return manifest


def _download_source(target: Path, source: SourceFile, revision: str) -> str:
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/"
        f"{urllib.parse.quote(source.path, safe='/')}?download=true"
    )
    temporary = target.with_name(target.name + ".part")
    temporary.unlink(missing_ok=True)
    attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1()
        git_sha1.update(f"blob {source.size}\0".encode("ascii"))
        byte_count = 0
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "cross-layer-5g-anomaly/TelecomTS-ingestion"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                with temporary.open("wb") as compressed_file:
                    with gzip.GzipFile(
                        filename="",
                        mode="wb",
                        fileobj=compressed_file,
                        compresslevel=6,
                        mtime=0,
                    ) as gzip_out:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            byte_count += len(chunk)
                            sha256.update(chunk)
                            git_sha1.update(chunk)
                            gzip_out.write(chunk)
            if byte_count != source.size:
                raise TelecomTSError(
                    f"Downloaded '{source.path}' has {byte_count} bytes; "
                    f"the Hub reports {source.size}."
                )
            if source.lfs_sha256 and sha256.hexdigest() != source.lfs_sha256:
                raise TelecomTSError(f"SHA-256 mismatch for '{source.path}'.")
            if not source.lfs_sha256 and git_sha1.hexdigest() != source.oid:
                raise TelecomTSError(f"Git blob ID mismatch for '{source.path}'.")
            os.replace(temporary, target)
            _LOGGER.info(
                "Downloaded %s (%.1f MB, compressed to %.1f MB).",
                source.path,
                byte_count / 1_000_000,
                target.stat().st_size / 1_000_000,
            )
            return sha256.hexdigest()
        except (OSError, urllib.error.URLError, TimeoutError, TelecomTSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                _LOGGER.warning(
                    "Download attempt %s/%s failed for %s: %s",
                    attempt,
                    attempts,
                    source.path,
                    exc,
                )
                time.sleep(attempt)
    raise TelecomTSError(
        f"Could not download verified TelecomTS source '{source.path}': {last_error}"
    ) from last_error


def _verify_source_archive(path: Path, source: SourceFile) -> bool:
    try:
        byte_count = 0
        digest = hashlib.sha256()
        git_sha1 = hashlib.sha1()
        git_sha1.update(f"blob {source.size}\0".encode("ascii"))
        with gzip.open(path, "rb") as source_in:
            while True:
                chunk = source_in.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
                git_sha1.update(chunk)
        if byte_count != source.size:
            return False
        if source.lfs_sha256:
            return digest.hexdigest() == source.lfs_sha256
        return git_sha1.hexdigest() == source.oid
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def _sha256_gzip_contents(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as source_in:
        while True:
            chunk = source_in.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_manifest(directory: Path, revision: str) -> dict[str, Any]:
    path = directory / "source_manifest.json"
    if not path.is_file():
        raise TelecomTSError(
            f"Offline mode requires an existing source manifest at '{path}'."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TelecomTSError(f"Could not read source manifest '{path}': {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("revision") != revision
        or not isinstance(manifest.get("files"), list)
    ):
        raise TelecomTSError("Source manifest does not match the requested dataset revision.")
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise TelecomTSError("Source manifest contains an invalid file entry.")
        source_path = item.get("path")
        expected_digest = item.get("downloaded_content_sha256")
        if not isinstance(source_path, str):
            raise TelecomTSError("Source manifest contains an invalid source path.")
        _validate_source_path(source_path)
        archive = directory / (source_path + ".gz")
        if not archive.is_file() or _sha256_gzip_contents(archive) != expected_digest:
            raise TelecomTSError(f"Offline source file is missing or changed: '{source_path}'.")
    return manifest


def _sample_id(
    record: Mapping[str, Any],
    revision: str,
    source_file: str,
    line_number: int,
) -> tuple[str, bool]:
    explicit = record.get("sample_id")
    if isinstance(explicit, (str, int)) and not isinstance(explicit, bool):
        return str(explicit), True
    stable_id = hashlib.sha256(
        f"{DATASET_ID}:{revision}:{source_file}:{line_number}".encode("utf-8")
    ).hexdigest()
    return f"telecomts:{stable_id}", False


def _valid_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _count_annotations(record: Mapping[str, Any], counters: Counter[str]) -> None:
    anomalies = record["annotations"]
    if not anomalies.get("exists"):
        return
    duration = anomalies.get("anomaly_duration")
    if (
        isinstance(duration, dict)
        and isinstance(duration.get("start"), int)
        and isinstance(duration.get("end"), int)
    ):
        counters["temporal_annotations"] += 1
    else:
        counters["missing_temporal_annotations"] += 1
    affected = anomalies.get("affected_kpis")
    if isinstance(affected, list) and any(
        isinstance(name, str) and name.strip() for name in affected
    ):
        counters["affected_kpi_annotations"] += 1
    else:
        counters["missing_affected_kpi_annotations"] += 1


def _collect_protocol_values(
    kpis: Mapping[str, Sequence[Any]],
    values_by_kpi: dict[str, set[str]],
) -> None:
    for name in CATEGORICAL_KPIS:
        values = kpis.get(name, ())
        values_by_kpi[name].update(value for value in values if isinstance(value, str))


def _get_json(url: str, label: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "cross-layer-5g-anomaly/TelecomTS-ingestion"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TelecomTSError(f"Could not retrieve {label}: {exc}") from exc


class _GzipTextWriter:
    """Small context wrapper to write deterministic UTF-8 gzip JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: BinaryIO | None = None
        self._gzip: gzip.GzipFile | None = None

    def __enter__(self) -> gzip.GzipFile:
        self._file = self.path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self._file,
            compresslevel=6,
            mtime=0,
        )
        return self._gzip

    def __exit__(self, *_exc: object) -> None:
        if self._gzip is not None:
            self._gzip.close()
        if self._file is not None:
            self._file.close()


def _gzip_writer(path: Path) -> _GzipTextWriter:
    return _GzipTextWriter(path)


def _write_json_line(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    stream.write(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _replace_file(temporary, path)


def _replace_file(source: Path, target: Path) -> None:
    os.replace(source, target)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _validate_revision(revision: str) -> None:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConfigurationError("TelecomTS revision must be a 40-character commit SHA.")


def _validate_source_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise TelecomTSError(f"Invalid source path in TelecomTS repository tree: {path!r}.")


def main(argv: Iterable[str] | None = None) -> int:
    """Download and ingest TelecomTS without fitting or evaluating models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Immutable 40-character Hugging Face dataset commit (default: pinned release).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Re-ingest verified local raw files without contacting Hugging Face.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root; defaults to the current working directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path.cwd().resolve() if args.project_root is None else args.project_root.resolve()
    report_log = root / "data" / "reports" / "telecomts" / args.revision / "ingestion.log"
    configure_logging("INFO", report_log)
    try:
        summary = ingest_telecomts(
            root,
            revision=args.revision,
            offline=args.offline,
        )
    except (ProjectError, OSError) as exc:
        _LOGGER.exception("TelecomTS ingestion failed: %s", exc)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
