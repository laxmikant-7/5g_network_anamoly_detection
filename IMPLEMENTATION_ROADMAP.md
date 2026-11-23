# Implementation Roadmap

## 1. Existing Architecture

### 1.1 Repository layout

```text
.
|-- backend/
|   |-- app.py
|   |-- deps.py
|   |-- main.py
|   `-- schemas.py
|-- frontend/
|   |-- api_client.py
|   |-- app.py
|   |-- components.py
|   |-- dashboard_data.py
|   `-- status.py
|-- src/
|   |-- common/
|   |   |-- exceptions.py
|   |   |-- logging_config.py
|   |   `-- paths.py
|   |-- data/
|   |   |-- schema.py
|   |   |-- split.py
|   |   |-- telecomts.py
|   |   |-- telecomts_preprocessing.py
|   |   `-- telecomts_eda.py
|   |-- evaluation/
|   |   `-- metrics.py
|   |-- features/
|   |   `-- builder.py
|   |-- models/
|   |   |-- diagnosis.py
|   |   `-- localization.py
|   `-- pipeline/
|       |-- capabilities.py
|       |-- config.py
|       |-- infer.py
|       |-- inference.py
|       `-- train.py
|-- tests/
|   |-- conftest.py
|   |-- helpers.py
|   |-- test_backend_api.py
|   |-- test_frontend_dashboard_data.py
|   |-- test_inference_pipeline.py
|   |-- test_telecomts_preprocessing.py
|   |-- test_telecomts_ingestion.py
|   `-- test_telecomts_eda.py
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- reports/telecomts/<revision>/eda/
|   |   |-- dataset_statistics.json
|   |   |-- data_quality.json
|   |   |-- preprocessing_config.json
|   |   `-- figures/*.png
|   `-- sample/schema_fixture.csv
|-- artifacts/
|-- docs/integration-test-report.md
|-- config.yaml
|-- pyproject.toml
`-- requirements.txt
```

`README.md` documents the current setup, TelecomTS ingestion, preprocessing, and
EDA. There are no production model files under `artifacts/` and no notebooks
implementing training.

### 1.2 Implemented modules

| Module | Existing responsibility |
|---|---|
| `src/common/paths.py` | Project-root discovery and safe project-relative path resolution. |
| `src/common/exceptions.py` | Project, configuration, schema, and artifact exception types. |
| `src/common/logging_config.py` | Structured logger setup and optional file logging. |
| `src/data/schema.py` | Data-frame schema and controlled-vocabulary validation. |
| `src/data/telecomts.py` | Pinned-revision TelecomTS JSONL discovery/download, compressed raw archive, 128-step record validation, direct label mapping, derived sample IDs, sequence/metadata outputs, and error reporting. |
| `src/data/telecomts_preprocessing.py` | Fixed-vocabulary, stateless one-hot encoding of the two protocol channels; strict numeric/schema validation; leakage-safe model-input selection; and persisted preprocessing configuration. |
| `src/data/telecomts_eda.py` | Streaming TelecomTS quality checks, descriptive statistics/correlations, and ten reproducible Matplotlib visualizations. |
| `src/pipeline/config.py` | Strict typed YAML loading for project, schema, split, model, API, and logging settings. |
| `src/pipeline/inference.py` | The working shared saved-artifact inference pipeline, including sample validation, preprocessing, feature extraction, binary detection, optional diagnosis, optional attribution, optional temporal localization, and artifact presence reporting. |
| `src/pipeline/infer.py` | JSON sample CLI entry point for the shared inference pipeline. |
| `src/pipeline/capabilities.py` | Capability flags. Training and evaluation are false; inference and the existing partial feature layer are true. |
| `backend/main.py` | FastAPI app factory, route registration, CORS, and project exception handlers. |
| `backend/deps.py` | FastAPI dependency exposing the typed application configuration. |
| `backend/schemas.py` | Pydantic API request and response schemas. |
| `frontend/api_client.py` | Standard-library HTTP client for health, model info, prediction, and complete analysis. |
| `frontend/app.py` | Nine-page Streamlit dashboard with explicit unavailable-data states. |
| `frontend/components.py` | Shared theme, header, metric, empty-state, and Plotly helpers. |
| `frontend/dashboard_data.py` | Uploaded sequence parsing, historical CSV loading, numeric-column discovery, and evaluation calculations from supplied labels. |
| `frontend/status.py` | Artifact and implementation-status payload for dashboard status views. |

The saved-artifact inference implementation is contract-heavy and is not a placeholder:
it applies a saved preprocessor, validates feature JSON, preserves declared feature
order, extracts supported summary statistics, loads joblib estimators, and supports
SHAP-like attribution and interval or mask temporal artifacts.

### 1.3 Incomplete modules and stubs

| Module | Current behavior | Missing behavior |
|---|---|---|
| `src/pipeline/train.py` | Validates enabled task configuration and refuses to train. | TelecomTS loading, feature fitting, estimator fitting, preprocessing fitting, artifact serialization, and metadata generation. |
| `src/data/split.py` | Checks the timestamp column and raises `NotImplementedError`. | Chronological split or entity/record-aware split with leakage safeguards. |
| `src/features/builder.py` | Reserved interface only. | Shared training feature matrix and alignment with the inference feature contract. |
| `src/models/localization.py` | Reserved fit/predict interfaces only. | Estimator selection, fitting, probability handling, and persistence. |
| `src/models/diagnosis.py` | Reserved fit/predict interfaces only. | Estimator selection, fitting, class handling, and persistence. |
| `src/evaluation/metrics.py` | Reserved localization and diagnosis metric interfaces only. | Accuracy, macro-F1, per-class metrics, confusion matrices, and robust holdout reporting. |

Additional missing project capabilities:

- No complete 128-step, 18-KPI training feature transformer shared with inference.
- No genuine train/validation/test artifact generation.
- No fit-time preprocessing or feature-config artifact generation.
- No model calibration or probability validation on real data.
- No persisted genuine evaluation report or holdout metrics.
- No real production KPI attribution or temporal-localization training.

The existing `data/sample/schema_fixture.csv` is only a four-column schema fixture.
It is not a 128-by-18 KPI dataset and must not be used to train or evaluate models.

The TelecomTS Hub dataset card and live dataset-server schema were inspected.
The pinned repository revision is `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`.
The source is 33 `processed/chunked.jsonl` files totaling approximately 1.12 GB
uncompressed, with one JSON sample per line. Each record has `start_time`,
`end_time`, `sampling_rate`, nested 18-channel `KPIs`, `anomalies`, and
contextual `labels`; sample IDs are absent from the source schema. The actual
sample and category counts are generated by the ingestion run and are not
replaced with dataset-card estimates. The full pinned run loaded 32,000 samples
(30,765 normal and 1,235 anomalous), validated all 33 source files with zero
invalid samples, and found temporal and affected-KPI annotations for all 1,235
anomalous samples.

The new ingestion module archives the pinned raw JSONL sources as gzip, emits
one valid KPI sequence and one metadata row per sample into separate gzip JSONL
files, and writes a JSON summary plus a compressed row-level validation error
report. IDs are retained when source-provided; because the inspected TelecomTS
schema has none, stable IDs are derived from revision, source path, and line
number. Invalid records remain in raw files and are identified in the report.
The ingest command does not train models.

The preprocessing and EDA modules use only the ingested KPI sequences as
predictive inputs and metadata only for analysis. Protocols use one-hot columns
with explicit categories `""`, `"None"`, `"TCP"`, `"UDP"`, and a separate
missing sentinel; the encoded shape is 128-by-26. Numeric values remain
unscaled and are not imputed. Since this transform is stateless, no data is fit
outside a training partition. Any future learned preprocessing must be fit
after a train split. The existing numeric-only API contract is unchanged and
no TelecomTS detector or inference artifact has been created.

The EDA CLI writes `dataset_statistics.json`, `data_quality.json`,
`preprocessing_config.json`, and ten PNG figures under
`data/reports/telecomts/<revision>/eda/`. The report includes exact counts,
source and category distributions, KPI descriptive statistics and correlations,
annotation intervals, quality errors, and categorical domains. Plot histograms
use a deterministic every-16th-timestep sample; other aggregates use all valid
processed samples.

The full EDA pass validated 32,000 processed samples against metadata: zero
invalid or duplicate samples, zero missing/non-finite KPI values, and zero
ingestion errors across 33 source files. It found 1,235 temporal intervals
(mean duration 91.81 and median duration 121 steps). See
[EDA statistics](data/reports/telecomts/1b3a88a440f4922edfcd1e230e42c003c0e13bdf/eda/dataset_statistics.json)
and [EDA quality report](data/reports/telecomts/1b3a88a440f4922edfcd1e230e42c003c0e13bdf/eda/data_quality.json).

The full run observed all 11 original anomaly classes and mapped them directly
to the configured project taxonomy; no categories were merged. It loaded 30,765
normal and 1,235 anomalous samples, with 1,235 temporal intervals, 1,235
affected-KPI annotations, and zero invalid records across 33 files. Per-class
counts and the full source inventory are in
[README.md](./README.md) and the generated
[validation report](data/reports/telecomts/1b3a88a440f4922edfcd1e230e42c003c0e13bdf/validation.json).

### 1.4 API request and response contracts

#### Request

`SampleRequest` requires:

- `sample`: exactly 128 rows.
- Each row: exactly 18 finite numeric values.
- Pydantic rejects missing, categorical, NaN, and infinite values.

#### Responses

| Endpoint | Method | Existing response |
|---|---|---|
| `/` | GET | Service name and route index. |
| `/health` | GET | `HealthResponse`: status, model loading status, loaded components, and artifact presence. |
| `/predict` | POST | `PredictionResponse`: normal/anomaly status, anomaly probability, and model version. |
| `/analyze` | POST | `AnalysisResponse`: binary result plus optional predicted type, diagnosis confidence, important KPI list, temporal interval, and explanation. |
| `/model-info` | GET | `ModelInfoResponse`: model versions, supported anomaly categories, and feature configuration. The configured category values now use the direct TelecomTS taxonomy; response structure is unchanged. |

The FastAPI exception policy is:

- HTTP 422 for Pydantic or pipeline sample/schema validation.
- HTTP 503 for missing required artifacts.
- HTTP 500 for configuration, inference, and unexpected server errors.

Existing response names and keys should remain unchanged unless a future phase
finds a documented compatibility requirement.

### 1.5 Feature and artifact contracts

Configured input contract:

- 128 time steps.
- 18 KPI channels.
- Channel names are declared in the feature JSON `kpi_names`.
- Feature definitions contain `name`, `kpi`, and `statistic`.
- Supported statistics are `mean`, `std`, `min`, `max`, `median`, `first`, `last`, and `slope`.

Required saved artifacts:

| Artifact | Config key | Current contract |
|---|---|---|
| Binary detector | `inference.detection_artifact` | Joblib object with `classes_`, `predict()`, and `predict_proba()`; exactly two classes with one identifiable anomaly class. |
| Preprocessor | `inference.preprocessing_artifact` | Joblib object with `transform()`; must preserve 128-by-18 shape and return finite numeric values. |
| Feature configuration | `inference.feature_config_artifact` | JSON object containing unique `kpi_names`, `feature_order`, and matching `features` definitions. |
| Model metadata | `inference.metadata_artifact` | JSON object with non-empty `model_version`; optional `model_versions` mapping. |

These are the existing numeric-only inference artifact contracts. The
TelecomTS preprocessing contract separately converts raw 18-channel sequences
to 26 numeric columns (16 unchanged numeric channels and ten one-hot columns).
It is not wired to or substituted for the current 128-by-18 API contract; that
integration remains for a later versioned API/model phase.

Optional artifacts:

| Artifact | Contract |
|---|---|
| Diagnosis | Joblib classifier or compatible estimator used only for anomalous samples. |
| Attribution | Joblib explainer with `shap_values(features)` or callable-compatible behavior; SHAP values must map to one value per configured feature. |
| Temporal localization | Joblib localizer implementing `predict_interval(processed)` or `predict(...)`; output is an inclusive start/end interval or a 128-element mask. |
| Localization | Configured separately by `config.yaml`, but no working training or inference interface exists yet. |

No trained artifact is currently available. Test-only joblib objects exist only in
temporary pytest directories and must never be treated as project artifacts.

### 1.6 Streamlit-to-FastAPI communication

- Streamlit loads `config.yaml` and derives the API origin as
  `http://<api_host>:<api_port>`.
- The configurable API URL is stored in Streamlit session state.
- `/health` is fetched for service status.
- `/model-info` is fetched for saved feature names and model versions.
- `POST /predict` receives `{"sample": [[...]]}`.
- `POST /analyze` receives the same request shape.
- Frontend errors are translated to `ApiClientError` messages.
- Streamlit does not maintain a second prediction implementation.
- Dataset exploration and evaluation use uploaded or local historical CSV data;
  they do not call a model or generate performance numbers.

### 1.7 Existing tests

The initial application suite contained 37 passing tests. TelecomTS ingestion
adds 18 tests; preprocessing and EDA add 12 more. The current full suite
contains 67 passing tests:

- `tests/test_backend_api.py`: 14 tests covering routes, Pydantic shape checks,
  missing artifacts, loaded state, CORS, and test-double inference.
- `tests/test_inference_pipeline.py`: 8 tests covering direct pipeline behavior,
  preprocessor application, feature order, missing optional artifacts, and
  unsupported categories.
- `tests/test_frontend_dashboard_data.py`: 15 tests covering sequence parsing,
  evaluation calculations, API connection failures, and Streamlit page rendering.
- `tests/test_telecomts_ingestion.py`: 18 tests covering label identity mapping,
  sequence/schema checks, annotations, invalid records, stable IDs, and repeatability.
- `tests/test_telecomts_preprocessing.py`: tests for the 26-channel one-hot
  contract, distinct protocol values, invalid inputs, metadata exclusion, config
  persistence, and serialized transform parity.
- `tests/test_telecomts_eda.py`: end-to-end report/figure generation and explicit
  sequence/metadata mismatch reporting.

The test doubles validate wiring and contracts. They do not validate real model
quality, calibration, telecom generalization, or genuine KPI attribution.

### 1.8 Configuration and dependencies

- `config.yaml`: project paths, the 12-label TelecomTS taxonomy, split fractions,
  artifact paths, task configuration, API address, and logging level.
- `pyproject.toml`: package mapping, console scripts `cl5g-train`, `cl5g-infer`,
  `cl5g-ingest-telecomts`, and `cl5g-telecomts-eda`, pytest settings, and runtime
  dependencies.
- `requirements.txt`: NumPy, pandas, scikit-learn, plotting libraries, FastAPI,
  Streamlit, SHAP, joblib, YAML, HTTPX, and pytest.
- `.gitignore`: excludes model files and raw/processed CSV contents while keeping
  `.gitkeep` placeholders.

## 2. Missing Components

1. Complete 128-step, 18-KPI training feature transformation.
2. Chronological split implementation with leakage prevention.
3. Detection, localization, and diagnosis training implementations.
4. Estimator registry and deterministic hyperparameter or baseline policy.
5. Fit-time preprocessing and pipeline serialization.
6. Holdout and cross-validation evaluation for all implemented tasks.
7. Genuine model metadata, feature configuration, and artifact serialization.
8. KPI attribution artifact generation and validation.
9. Temporal anomaly localizer training and artifact serialization.
10. Evaluation report artifact and Streamlit loading contract.
11. Expanded training and model tests using approved real or fixture data.

## 3. Files to Create

Creation is expected only after the implementation phases below begin. The names
are architectural candidates and may be consolidated if repository review finds
a clearer existing-module fit.

| File | Purpose |
|---|---|
| `src/features/extraction.py` | Complete sequence feature extraction used by both training and inference. |
| `src/models/registry.py` | Named estimator construction and task-specific configuration. |
| `src/models/temporal.py` | Real temporal-localization model fit/predict interface. |
| `src/evaluation/report.py` | JSON/CSV holdout report generation and artifact serialization. |
| `src/pipeline/artifacts.py` | Shared artifact loading, manifest writing, compatibility checks, and atomic persistence. |
| `src/pipeline/train_detection.py` | Binary detection training orchestration. |
| `src/pipeline/evaluate.py` | CLI entry point for held-out genuine evaluation. |
| `tests/test_feature_builder.py` | Feature consistency, leakage, and training/inference parity tests. |
| `tests/test_split.py` | Time/entity split and leakage tests. |
| `tests/test_training.py` | Training orchestration and artifact contract tests. |
| `tests/test_evaluation.py` | Genuine metric and report tests. |
| `tests/test_temporal.py` | Temporal interval/mask contract tests. |
| `artifacts/model_manifest.json` | Generated manifest only after a genuine trained run; never a planning stub or fake metric. |
| `artifacts/evaluation_report.json` | Generated genuine holdout report only after real ground truth and predictions exist. |

Do not add synthetic TelecomTS data or synthetic trained model binaries as substitutes
for approved source data.

## 4. Files to Modify

| File | Required future modification |
|---|---|
| `src/data/schema.py` | Add normalized training-table validation if the upcoming training interface requires it; TelecomTS-specific record validation now lives in `src/data/telecomts.py`. |
| `src/data/split.py` | Replace reserved splitter with a time-ordered implementation. |
| `src/features/builder.py` | Implement the shared training feature matrix, or delegate to `src/features/extraction.py` without duplicating inference logic. |
| `src/models/localization.py` | Implement fit/predict and saved-estimator behavior. |
| `src/models/diagnosis.py` | Implement fit/predict and saved-estimator behavior. |
| `src/evaluation/metrics.py` | Implement genuine localization accuracy and diagnosis macro-F1 and per-class metrics. |
| `src/pipeline/train.py` | Replace refusal behavior with a configurable training pipeline after dependencies are complete. |
| `src/pipeline/inference.py` | Consume the finalized shared feature contract and add artifact manifest compatibility checks without changing API response keys. |
| `src/pipeline/capabilities.py` | Set capability flags only after real implementation and tests exist. |
| `src/pipeline/config.py` | Add approved training, telemetry, and artifact fields only where technically required. |
| `backend/main.py` | Add routes or fields only if needed; preserve existing routes and response shapes. |
| `backend/schemas.py` | Extend schemas only through backward-compatible optional fields or separately versioned endpoints. |
| `frontend/app.py` | Load persisted genuine evaluation artifacts and integrate new pages/features only after backend contracts are stable. |
| `frontend/dashboard_data.py` | Add real evaluation artifact loaders while retaining user-supplied evaluation support. |
| `frontend/status.py` | Report training/evaluation status from capability flags and actual artifacts. |
| `config.yaml` | The anomaly taxonomy now matches the direct TelecomTS mapping; add model/training settings only after feature and artifact contracts are finalized. |
| `pyproject.toml` | Contains the TelecomTS ingestion and EDA entry points; update future scripts/dependencies only if required. |
| `requirements.txt` | Keep duplicate dependency declarations synchronized with `pyproject.toml`. |
| `docs/integration-test-report.md` | Add separate genuine-data/evaluation evidence; do not mix it with test-double evidence. |

## 5. Integration Dependencies

1. Real or approved TelecomTS-compatible data must precede model training.
2. Dataset timestamp, entity, layer, anomaly-type, and KPI column semantics must be
   confirmed before ingestion code is finalized.
3. The shared feature specification must be implemented once and reused by training
   and inference.
4. The chronological split must precede preprocessing fitting and estimator training.
   The current TelecomTS encoder is stateless; any learned transforms must only be
   added after this split is implemented.
5. Preprocessing and feature configuration artifacts must be produced together.
6. Detection training must precede diagnosis, localization, and optional explanation
   work when the pipeline is anomaly-gated.
7. Artifact metadata must include package/model/feature compatibility and genuine
   training run identity.
8. Backend loading must validate artifact compatibility before serving predictions.
9. Streamlit must continue consuming backend JSON and must not replicate model logic.
10. A genuine evaluation report must exist before the dashboard can display persisted
    validated model performance.

## 6. Expected Inputs and Outputs of New Modules

### TelecomTS ingestion

- Input: all pinned-revision Hugging Face `processed/chunked.jsonl` files or
  their verified local compressed archives.
- Output: `sequences.jsonl.gz` containing the 128-step, 18-channel original KPI
  arrays; `metadata.jsonl.gz` containing IDs, source path/line, times, binary
  label, mapped anomaly type, and source annotations.
- Validation: exact KPI names and sequence lengths; numeric finite values;
  categorical string domain; timestamps; sampling rate; anomaly label and
  presence agreement; inclusive interval bounds; affected KPI names; duplicate
  IDs; malformed JSON/UTF-8; and a persistent row-level error report.
- Reproducibility: immutable revision, SHA/LFS-verified source manifest,
  deterministic source-derived IDs, atomic replace of outputs, and offline
  repeat-ingestion mode.

### TelecomTS preprocessing

- Input: one processed sequence record containing only the KPI mapping as
  predictive data; labels and other metadata are ignored by the encoder.
- Output: a numeric 128-by-26 matrix with 16 original numeric channels and five
  one-hot columns for each protocol KPI.
- Categories: `""`, `"None"`, `"TCP"`, `"UDP"`, and a separate null/missing
  sentinel. The empty string and literal `"None"` remain distinct; unknown
  categories and missing/non-finite numeric values are rejected.
- Fitting: current encoder has no learned state, scaling, clipping, or imputation.
  Any future learned transformation must be fitted on training data only.
- Reuse: the stateless, serializable transformer and JSON configuration define
  the same transform for training and future inference; the existing 128-by-18
  REST endpoint remains unchanged pending API versioning and model artifacts.

### TelecomTS EDA and data quality

- Input: the pinned revision's processed sequences and metadata plus ingestion
  validation/error reports.
- Output: `dataset_statistics.json`, `data_quality.json`,
  `preprocessing_config.json`, and ten Matplotlib PNGs in the revision-specific
  `eda/` directory.
- Validation: paired row counts and IDs, uniqueness, KPI names and dimensions,
  numeric/categorical value domains, labels, binary label consistency, temporal
  intervals, affected-KPI annotations, missing values, and non-finite values.
- Analysis: counts, numeric descriptive statistics and correlations, protocol
  domains, source distribution, representative time-series, and inclusive
  anomaly interval distributions. Predictive model quality is not measured.

### Chronological splitting

- Input: normalized labeled table and split fractions.
- Output: unambiguous train, validation, and test row partitions or entity-aware
  partitions where entity leakage would otherwise occur.
- No output from this module may be shuffled across time.

### Feature extraction

- Input: normalized 128-step KPI sequence and feature JSON.
- Output: one pandas feature row, or a table of feature rows for training.
- Contract: names and order must exactly match the persisted inference
  configuration.

### Detection training

- Input: training feature matrix and binary labels.
- Output: fitted detector, fitted preprocessing object, feature JSON, model
  metadata, and training run diagnostics.

### Localization and diagnosis training

- Input: task-specific training features and held-out-compatible labels.
- Output: fitted estimator and metadata with supported class/layer names.

### Evaluation report

- Input: held-out ground truth and predictions for each implemented task.
- Output: JSON and/or CSV metrics, confusion matrices, per-class metrics,
  class distributions, split identity, and data provenance.

### Artifact persistence

- Input: trained objects and training report.
- Output: joblib/JSON files plus a manifest linking versions and compatibility.
- No output may contain invented metrics or placeholder model quality.

### Temporal and attribution training

- Input: approved time-window anomaly labels and real feature or sequence data.
- Output: only a model artifact and validation evidence supported by ground truth.
- Unsupported or unavailable adapters remain explicitly absent.

## 7. Testing Strategy

### Baseline

- Run `python -m pytest` before each implementation phase.
- The original 37 application tests and all 18 ingestion tests remain passing;
  the full suite now includes 12 preprocessing/EDA tests (67 total).

### Data and split tests

- Add tests for real or approved fixture formats.
- Verify required columns, timestamps, entities, layers, labels, numeric values,
  and format rejection.
- Verify the validation partition occurs before or between training periods as
  designed, without cross-split row or entity leakage.

### Feature tests

- Verify every configured statistic against a simple known sequence.
- Verify training and inference produce identical names and values.
- Verify no training-only values leak into inference preprocessing.

### Training tests

- Use only approved fixture data or real TelecomTS-derived test data.
- Assert artifact schemas, no in-request fitting, deterministic IDs where
  applicable, and supported class lists.
- Never assert that test-generated data proves production quality.

### Evaluation tests

- Test known labels and probabilities against independently calculated metrics.
- Verify macro-F1 and localization accuracy are omitted or marked unavailable
  until genuine held-out labels exist.
- Test empty predictions, class imbalance, missing classes, and invalid scores.

### Artifact and API tests

- Preserve `/health`, `/predict`, `/analyze`, `/model-info`, response keys, and
  HTTP 422/503/500 behavior.
- Test real model artifact loading separately from test doubles.
- Test missing, corrupt, incompatible, and unsupported artifacts.

### Dashboard tests

- Render all nine pages.
- Keep unavailable performance as unavailable.
- Verify persisted genuine evaluation data is labeled as genuine held-out or
  test output and never inferred from live predictions.

## 8. Implementation Order

1. Pin and download the real TelecomTS source (complete).
2. Complete full TelecomTS ingestion and review its validation report (complete;
   32,000 records loaded, zero invalid).
3. Implement and test chronological/entity-safe splitting.
4. Extract and test complete features shared by training and inference.
5. Implement estimator construction and binary detection training.
6. Fit preprocessing, detection, feature JSON, and metadata artifacts.
7. Implement genuine held-out binary evaluation and an evaluation report.
8. Implement diagnosis training and diagnosis artifact serialization.
9. Implement localization training if approved TelecomTS labels support it.
10. Implement and validate attribution generation only if a supported explainability
    method and model interface are available.
11. Implement and validate temporal localization training only if ground truth
    supports time intervals or anomaly masks.
12. Integrate new artifacts into the existing shared inference pipeline.
13. Update FastAPI only if technically required and preserve existing contracts.
14. Update Streamlit status/evaluation views while preserving all nine pages.
15. Run complete tests and create a separate genuine-data integration report.

No implementation task in this roadmap should generate or serialize a model
artifact without approved source data and a verifiable training/evaluation run.

