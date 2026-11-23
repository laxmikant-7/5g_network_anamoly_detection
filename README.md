# Cross-Layer 5G Anomaly Localization and Diagnosis

This repository contains a FastAPI inference service, a nine-page Streamlit
dashboard, and a pinned TelecomTS ingestion pipeline. Dataset ingestion is
separate from model training: running the ingestion command does not fit,
evaluate, or serialize any model.

## TelecomTS dataset

Source: [AliMaatouk/TelecomTS on Hugging Face](https://huggingface.co/datasets/AliMaatouk/TelecomTS)  
Dataset implementation and schema notes:
[Ali-maatouk/TelecomTS on GitHub](https://github.com/Ali-maatouk/TelecomTS)

The ingester defaults to immutable Hub commit
`1b3a88a440f4922edfcd1e230e42c003c0e13bdf`. It discovers all
`*/processed/chunked.jsonl` files from that revision and downloads each file
only once. The current Hub tree lists 33 source files totaling about 1.12 GB
uncompressed; local raw copies are gzip-compressed. Check available disk space
before the first run.

### Ingest

From the repository root, with the project dependencies installed:

```powershell
python -m cl5g.data.telecomts
```

By default, the command fetches and verifies each source file against its
Hugging Face Git/LFS digest, then ingests the local archive. To re-run
validation and processing without network access:

```powershell
python -m cl5g.data.telecomts --offline
```

Use `--revision <40-character-commit-sha>` to explicitly ingest another
immutable dataset version. Ingestion is repeatable: processed outputs are
replaced atomically rather than appended. It emits progress/errors to the
console and writes a persistent ingestion log.

### Storage layout

Raw downloads and processed sequence data are ignored by Git; validation and
EDA reports are kept under the revision-specific reports directory:

```text
data/
|-- raw/telecomts/<revision>/
|   |-- <original repository paths>/chunked.jsonl.gz
|   `-- source_manifest.json
|-- processed/telecomts/<revision>/
|   |-- sequences.jsonl.gz
|   `-- metadata.jsonl.gz
`-- reports/telecomts/<revision>/
    |-- validation.json
    |-- invalid_records.jsonl.gz
    |-- ingestion.log
    `-- eda/
        |-- dataset_statistics.json
        |-- data_quality.json
        |-- preprocessing_config.json
        `-- figures/
```

`sequences.jsonl.gz` contains one valid sample per line, with a stable
`sample_id` and all original ordered KPI arrays. The source has no sample ID in
its documented schema; when absent, the ingester creates a stable SHA-256 ID
from the pinned revision, source path, and source line. `metadata.jsonl.gz`
contains that ID, source-file path and line, source/normalized labels, times,
and the original anomaly and contextual-label annotations. The compressed raw
JSONL archives retain the unmodified complete records, including descriptions,
statistics, and Q&A.

The validation report includes source-file/sample counts, accepted and invalid
record counts, category distributions, categorical value inventories,
annotation coverage, and the paths/digests used. Every invalid record is
represented in `invalid_records.jsonl.gz` by source file, source line, stable
sample ID when parseable, and validation errors. It remains available in the
raw archive and is not quietly dropped from the audit trail.

## Preprocessing and exploratory analysis

Run analysis on the ingested revision:

```powershell
python -m cl5g.data.telecomts_eda
```

Or, after installing the project, run `cl5g-telecomts-eda`. The command is
offline and reads the processed sequences, metadata, and ingestion reports.
Outputs are written to `data/reports/telecomts/<revision>/eda/`:

- `dataset_statistics.json`: label and source-file distributions, numerical
  KPI statistics/correlations, protocol values, and temporal interval summaries.
- `data_quality.json`: sequence/schema/label/annotation checks, duplicate IDs,
  missing/non-finite values, source alignment, and ingestion error counts.
- `preprocessing_config.json`: versioned input/output order, category vocabulary,
  missing/unknown value policy, and leakage exclusions.
- `figures/`: ten PNGs for binary and per-category distributions, numeric
  distributions/correlations, representative time series, normal/anomalous
  comparisons, source files, temporal intervals, missingness, and protocols.

### Categorical representation and leakage controls

At the pinned revision each protocol KPI contains four observed string values:
the empty string, literal `"None"`, `"TCP"`, and `"UDP"`. The
`TelecomTSPreprocessor` uses explicit one-hot columns for those values and a
fifth `__MISSING__` column for an actual JSON null at transform time. The empty
string and literal `"None"` remain distinct; neither is treated as an ordered
or continuous numeric code. Unknown strings are rejected. Numeric KPI values
remain in their raw units; missing, non-numeric, and non-finite numeric values
are rejected and reported rather than imputed or clipped. This avoids
changing anomaly patterns. The encoded sequence has shape 128-by-26: 16 numeric
channels plus five one-hot columns for each categorical KPI.

The encoder is stateless and its fixed vocabulary/configuration is persisted,
so the same serializable `TelecomTSPreprocessor` transform can be reused by
training and inference; tests verify identical output after serialization. No
scaling, imputation, or other learned transform is fitted. Any future learned
transform must be fitted only after a training split is established. Only
`KPIs` are predictive input. Labels, anomaly descriptions and intervals,
affected KPI annotations, source identifiers, timestamps, and
zone/application/mobility/congestion metadata are retained for analysis but
excluded from the encoder.

## Source-file-aware train/validation/test split

Create reproducible TelecomTS partitions with:

```powershell
python -m cl5g.data.split
```

Or, after installation, run `cl5g-split-telecomts`. The configured default is
70% training, 10% validation, and 20% test, with seed `42`. Nested
`StratifiedGroupKFold` candidates and a seeded `GroupShuffleSplit` validation
search keep every source JSONL file entirely within one partition. Candidate
selection favors feasible class support and then the requested sample and
class proportions; it never splits a source group to force balance. Use
validation for feature/model selection and threshold tuning. The final test
partition is not for model-selection decisions.

The command writes paired compressed sequence and metadata files plus
`split_metadata.json` under
`data/processed/telecomts/<revision>/splits/seed-<seed>/`. Split rows retain
their original sample IDs, source IDs, labels, metadata, and KPI sequences.
The metadata report includes per-partition sample/normal/anomaly and class
counts, source-file lists, overlap checks, class support, requested/achieved
fractions, and anomaly categories absent from validation or test. If the
dataset had previously been used for test evaluation, pass
`--historical-test-data-used`; the report will mark that test partition as
non-pristine. No earlier model or evaluation artifacts were present in this
workspace when this split was generated.

Pinned-revision split result (seed `42`):

| Partition | Samples | Source files | Normal | Anomalous |
|---|---:|---:|---:|---:|
| Training | 22,415 | 24 | 21,746 | 669 |
| Validation | 3,233 | 4 | 3,006 | 227 |
| Test | 6,352 | 5 | 6,013 | 339 |

No source files overlap. All 11 anomaly categories are represented in both
validation and test; exact per-category support is in `split_metadata.json`.
The 32,000 original sample IDs and sequence/metadata pairs were accounted for
without duplication or content changes.

## Observed TelecomTS schema

The Hub dataset card and live dataset-server schema were inspected before
implementing the reader. Each JSONL line is a sample object with `start_time`,
`end_time`, `sampling_rate`, `KPIs`, `description`, `anomalies`, `statistics`,
`labels`, and `QnA`. `KPIs` is a dictionary of 18 128-element lists. The
anomaly object contains `exists`, `type`, `anomaly_duration` (`start`, `end`),
and `affected_kpis`; normal records use false/empty/null annotations. Context
labels include `anomaly_present` as `"Yes"` or `"No"`.

The verified 16 numeric KPI names are:

`RSRP`, `DL_BLER`, `DL_MCS`, `UL_BLER`, `UL_MCS`, `UL_NPRB`, `UL_SNR`,
`TX_Bytes`, `RX_Bytes`, `Estimated_UL_Buffer`, `PRBs_DL_Current`,
`PRBs_UL_Current`, `PRB_Utilization_DL`, `PRB_Utilization_UL`,
`UL_NumberOfPackets`, `DL_NumberOfPackets`.

The two categorical KPI names are `UL_Protocol` and `DL_Protocol`; values are
strings, not numeric encodings. The inspected records include `TCP`, `UDP`,
`None`, and the empty string. Empty string is preserved as the dataset's
no-protocol categorical value. The ingestion validator rejects JSON null and
non-string source values; the reusable preprocessing encoder reserves a
separate missing category for nulls supplied directly at transform time.
The ingestion report records the full observed domain for each categorical
channel at the pinned revision.

The original dataset's 11 anomaly classes map directly, without merging, to
the project taxonomy:

| Dataset anomaly type | Project label |
|---|---|
| Jamming | Jamming |
| Co-Channel Interference (Mild) | Co-Channel Interference (Mild) |
| Buffer Overflow (Gradual Buildup) | Buffer Overflow (Gradual Buildup) |
| Co-Channel Interference (Severe) | Co-Channel Interference (Severe) |
| Antenna Failure | Antenna Failure |
| Faulty RF Filters (Temporal) | Faulty RF Filters (Temporal) |
| High Network Congestion (Gradual Buildup) | High Network Congestion (Gradual Buildup) |
| Doppler Shift (Severe) | Doppler Shift (Severe) |
| Faulty Handover Algorithm (Too Frequent) | Faulty Handover Algorithm (Too Frequent) |
| Resource Allocation Bugs | Resource Allocation Bugs |
| High Network Congestion (Sudden Spike) | High Network Congestion (Sudden Spike) |
| `exists=false` | Normal |

The source schema documents no sample ID field. Source-file identifiers are
the original repository-relative JSONL paths. Temporal intervals are preserved
as inclusive source indices (0 through 127), and affected KPI names are kept
as supplied.

The Hub tree groups the 33 processed JSONL sources as follows:

```text
anomalous/jammer/{File,Twitch,YouTube}/processed/chunked.jsonl
anomalous/synthetic/Zone_{A,B,C}/{File,Twitch,YouTube}/processed/chunked.jsonl
normal/mobile/{File,Twitch,YouTube}/processed/chunked.jsonl
normal/stationary/Zone_{A,B,C}/{congestion,no_congestion}/{File,Twitch,YouTube}/processed/chunked.jsonl
```

The source directory names are not used to infer sample labels; labels are
validated from each record's `anomalies.exists`, `anomalies.type`, and
`labels.anomaly_present` fields.

## Shared feature extraction

The training feature builder and saved-artifact inference pipeline use the same
ordered implementation in `cl5g.features.extraction`. On TelecomTS samples it
consumes the original 128-by-18 KPI order, applies the fixed stateless
one-hot protocol encoder, then extracts features from the resulting 128-by-26
numeric sequence. The API accepts the observed protocol strings only in their
corresponding categorical KPI channels; legacy numeric 128-by-18 requests and
feature configurations remain supported.

The planned feature experiments are generated without fitting or training:

```powershell
python -m cl5g.features.extraction
```

Or, after installing the project, run `cl5g-telecomts-features`. The command
reads the pinned processed TelecomTS sample and writes revision-specific
artifacts under `data/reports/telecomts/<revision>/features/`:

- `feature_configs/`: ordered inference-compatible JSON configurations for
  base, temporal, distribution, mean-absolute-difference, and selected
  combinations.
- `feature_dimensions.json`: input/processed shapes and every feature name,
  order, and count.
- `feature_example.json`: shared extraction from a real sample, with its label
  explicitly marked as analysis-only.

For each of the 16 numerical KPIs, base statistics are mean, population
standard deviation, minimum, maximum, median, first, and last value. Temporal
statistics are centered slope, first/second-half means, mean absolute
difference, root mean square, skewness, excess kurtosis, and interquartile
range. Each categorical KPI contributes one frequency per configured protocol
category, not a numeric category code. Constant sequences produce finite
zero-valued dispersion/shape statistics. Inputs and outputs are checked for
shape, category encoding, missing values, and non-finite values.

Feature reports can be generated without running ingestion or EDA if the
revision's processed `sequences.jsonl.gz` and `metadata.jsonl.gz` already
exist. This phase only defines and validates features; it does not fit a
classifier. The training workflow below consumes the same shared feature
builder and creates the detector artifacts.

## Binary detection training and evaluation

Train the binary anomaly detector from the saved source-file-grouped split:

```powershell
python -m cl5g.pipeline.train
```

Or, after installing the project, run `cl5g-train`. The training command
compares a prior-only `DummyClassifier`, unweighted and class-weighted Random
Forests, and (when installed) unweighted and weighted XGBoost. It uses the
`base_plus_temporal` 250-feature configuration shared with inference. Feature
construction is batched, and no dataset labels or metadata are passed to the
feature builder.

All candidates are fitted on training data. Candidate selection and F2
threshold optimization use validation only; the F2 tie-breaks prefer higher
recall, then precision, lower false-positive rate, and higher threshold. The
selected estimator is refit on the training partition, then the saved model
and selected threshold are evaluated once on the final test partition. The
test-evaluation state file prevents accidentally running that final evaluation
again. SMOTE is not used because interpolating categorical-frequency and
time-series summary features may create implausible samples. Class weighting
and threshold selection are compared instead.

The detector is saved at `artifacts/detection.joblib`, with its stateless
preprocessor at `artifacts/preprocessing.joblib`, ordered feature contract at
`artifacts/feature_config.json`, and model version/decision threshold at
`artifacts/model_metadata.json`. The model accepts the feature names in the
saved configuration and uses the positive-class probability with the saved
`decision_threshold`. Older model metadata without that field continues to
use the estimator's `predict()` result.

The real run for dataset revision
`1b3a88a440f4922edfcd1e230e42c003c0e13bdf`, seed `42`, selected the
unweighted Random Forest using validation F2. Its threshold was `0.429333`,
selected on the 3,233-sample validation partition. The final test partition
was marked pristine by the split metadata at that time and was evaluated once
for binary detection. It has since also been used for diagnosis evaluation;
it is no longer an untouched test set:

| Test measure | Result |
|---|---:|
| Samples (normal / anomalous) | 6,352 (6,013 / 339) |
| Accuracy | 0.9690 |
| Anomaly precision | 0.7064 |
| Anomaly recall | 0.7168 |
| Anomaly F1 / F2 | 0.7116 / 0.7147 |
| Average precision | 0.7825 |
| ROC-AUC | 0.9500 |
| False-positive rate | 0.0168 |
| Confusion matrix (normal/anomaly rows and columns) | `[[5912, 101], [96, 243]]` |

All 11 anomaly categories have test support. Category-level recall and support,
the classification report, and false-positive/false-negative sample IDs are
in the test report directory. In particular, the model detected none of the
14 `High Network Congestion (Sudden Spike)` test samples; the overall metrics
should therefore not be interpreted as uniform category performance.

The revision-specific report directory
`data/reports/telecomts/<revision>/binary_detection/seed-<seed>/` contains
validation candidate comparisons, validation threshold CSV/plot, final test
metrics, classification report, error analysis, confusion matrix, and
precision-recall and ROC curves. It also records the one-time evaluation
state. The candidate results show why accuracy alone is insufficient: the
prior-only baseline obtains 100% anomaly recall only by labeling every sample
anomalous (100% false-positive rate).

## Multiclass anomaly diagnosis

Train the anomaly-type classifier from anomalous samples in the same
source-file-grouped split:

```powershell
python -m cl5g.pipeline.train_diagnosis
```

Or, after installing the project, run `cl5g-train-diagnosis`. The training
partition supplies fit data; validation anomalies select among unweighted and
class-balanced Random Forest, class-balanced Extra Trees, and (when installed)
class-balanced XGBoost candidates. Candidate selection maximizes validation
macro-F1 over classes with actual validation support, with macro-recall,
balanced-accuracy, and weighted-F1 tie-breaks. No oversampling or synthetic
training rows are used. Evaluation metrics and confusion matrices are based
only on anomaly samples. Every original anomaly label is checked against the
ingester's source-label mapping; the 11 names map exactly, with no merging.

The model, an inference-compatible string-label wrapper, and its class mapping
are saved to `artifacts/diagnosis.joblib` and
`artifacts/diagnosis_class_mapping.json`. `artifacts/diagnosis_metadata.json`
and `artifacts/diagnosis_configuration.json` record the pinned data revision,
feature ordering, training class support, seed, selected estimator, and
configuration. The existing inference pipeline uses the configured diagnosis
artifact only after binary detection returns an anomaly; normal predictions
return without loading or running diagnosis.

### Measured diagnosis results

For revision `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`, seed `42`, the
balanced Random Forest won validation selection. The model was fit on 669
actual anomalous training samples; validation used 227 anomalous samples and
the test evaluation used 339 anomalous samples. All 11 categories were
represented in training, validation, and test.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.7137 | 0.4572 |
| Balanced accuracy | 0.5723 | 0.2626 |
| Macro precision | 0.5850 | 0.3799 |
| Macro recall | 0.5723 | 0.2626 |
| Macro F1 | 0.5575 | 0.2326 |
| Weighted F1 | 0.7192 | 0.3587 |

Test-set class recall and support:

| Anomaly category | Train support | Test support | Test recall |
|---|---:|---:|---:|
| Jamming | 93 | 94 | 1.000 |
| Co-Channel Interference (Mild) | 100 | 41 | 0.439 |
| Buffer Overflow (Gradual Buildup) | 101 | 21 | 0.524 |
| Co-Channel Interference (Severe) | 49 | 58 | 0.069 |
| Antenna Failure | 67 | 33 | 0.818 |
| Faulty RF Filters (Temporal) | 66 | 26 | 0.038 |
| High Network Congestion (Gradual Buildup) | 33 | 11 | 0.000 |
| Doppler Shift (Severe) | 50 | 18 | 0.000 |
| Faulty Handover Algorithm (Too Frequent) | 48 | 13 | 0.000 |
| Resource Allocation Bugs | 39 | 10 | 0.000 |
| High Network Congestion (Sudden Spike) | 23 | 14 | 0.000 |

The largest test confusion was `Co-Channel Interference (Severe)` predicted as
`Co-Channel Interference (Mild)` (43 samples). Other major errors mapped
`Doppler Shift (Severe)` and `Faulty RF Filters (Temporal)` to `Jamming` (18
each). The Sudden Spike category has the smallest training support (23); the
training report flags counts below 25 as a review warning, not as a claim of a
formal statistical minimum. The poor test performance for several categories
despite moderate training counts is also recorded for follow-up.

The diagnosis test partition is **not pristine**: it was already used for
binary-detector evaluation. Diagnosis candidate and estimator selection used
validation only, and the test evaluation ran once for this diagnosis run, but
its metrics are a historical holdout estimate rather than a new untouched-test
claim. `data/reports/telecomts/<revision>/multiclass_diagnosis/seed-<seed>/`
contains supported-class validation/test metrics, per-class results, full
11-by-11 confusion matrices, confusion analysis, support and missing-class
reports, validation/test differences, and the one-time evaluation state.

The real-data end-to-end smoke report confirms a normal training sample returns
normal without loading diagnosis, while a binary-positive Jamming sample
continues to multiclass diagnosis and returns Jamming. These in-sample smoke
examples validate integration only and are not performance metrics.

## Temporal anomaly localization

Train the per-timestep interval localizer and generate its group-aware
evaluation report with:

```powershell
python -m cl5g.evaluation.temporal_localization
```

The source annotations are sample-specific inclusive indices in the original
128-step sequence. In the pinned TelecomTS revision, all 1,235 anomalous
samples have valid intervals and affected-KPI annotations; all 30,765 normal
samples have the source's empty-interval representation. The sampling rate is
10 Hz (0.1 seconds per step), and affected-KPI annotations cover 16 numerical
channels. However, 593 anomaly intervals cover all 128 steps and the median
annotated interval is 121 steps, so many annotations contain little boundary
information. The interval boundaries have not been independently checked
against raw event timing.

The experimental supervised model is a class-weighted Random Forest over
centered five-step contexts of the stateless 26-channel encoded KPI sequence.
It is trained on annotation-derived timestep labels from anomalous training
records only. Source-file groups remain disjoint (24 train, 4 validation, 5
test files); the probability threshold is selected on validation by mean
temporal IoU. Inference passes the original time-aligned KPI sequence to this
adapter rather than the sample-level feature vector. Normal predictions skip
localization.

The model was selected at threshold `0.42`. Validation mean IoU was `0.8144`
versus `0.7592` for the full-window heuristic baseline. On the group-disjoint
test partition, model mean IoU was `0.7606` versus `0.7664` for that baseline;
the model therefore does **not** demonstrate improved test generalization over
the trivial baseline. The test partition is historical: binary detection and
diagnosis have already used it, and it was not used for localization threshold
selection. Treat the current localizer as experimental, not as a validated
operational model.

| Partition / method | Mean IoU | Event P / R (IoU >= 0.5) | Boundary MAE (steps) | Timestep P / R / F1 |
|---|---:|---:|---:|---:|
| Validation / supervised | 0.8144 | 0.8133 / 0.8062 | 12.09 | 0.8239 / 0.9703 / 0.8911 |
| Validation / full-window baseline | 0.7592 | 0.7313 / 0.7313 | 15.41 | 0.7592 / 1.0000 / 0.8631 |
| Test / supervised | 0.7606 | 0.7537 / 0.7493 | 15.51 | 0.7739 / 0.9916 / 0.8693 |
| Test / full-window baseline | 0.7664 | 0.7434 / 0.7434 | 14.95 | 0.7664 / 1.0000 / 0.8678 |

The report includes temporal IoU, event precision/recall at IoU `>= 0.5`,
boundary MAE, timestep precision/recall/F1, full-window baseline comparisons,
validation threshold curves, and a sample interval visualization. The
returned localization score is the mean timestep probability inside the
predicted interval; it is uncalibrated. Affected KPI predictions are null:
the model does not infer them from attribution scores.

`/analyze` accepts an optional `sampling_rate_hz` (defaults to the measured
TelecomTS rate of 10 Hz) and returns inclusive step boundaries plus start,
end, and duration in seconds. The Streamlit Temporal Localization page shows
the interval, rate, score calibration status, and experimental-model warning.
The artifact is `artifacts/temporal_localization.joblib`; the pinned run's
report, threshold data, and visualizations are under
`data/reports/telecomts/<revision>/temporal_localization/seed-<seed>/`.

## KPI attribution and explainability

Generate the global explainability reports and save the inference adapter with:

```powershell
python -m cl5g.evaluation.attribution
```

The report uses validation data only: 256 stratified validation samples for
binary detection and the 227 validation anomalies for diagnosis in the pinned
seed-42 run. It does not inspect the test partition. Global reports include
TreeSHAP mean-absolute feature importance and three-repeat permutation
importance (average precision for detection; macro-F1 over validation-supported
classes for diagnosis). For multiclass SHAP, absolute contributions are
averaged across samples and output classes.

Engineered-feature attribution maps back to the original KPI using the saved
feature configuration. KPI-level SHAP importance is the sum of absolute
contributions for that KPI's features; signed contributions are also summed
for local explanations. Permutation KPI scores sum only positive
feature-level score decreases; negative values remain visible in feature-level
reports. These scores indicate model associations, not physical causes.

The optional adapter is saved as `artifacts/attribution.joblib`. Inference
returns separate structured `attributions.binary_detection` and
`attributions.multiclass_diagnosis` objects for anomalous predictions, with
the predicted class, probability, baseline/output values, top feature
contributions, and ranked KPI scores. Normal predictions skip attribution;
when the adapter is unavailable, the response leaves attributions null. The
Streamlit KPI Attribution page displays the available model-specific scores.

The revision-specific output directory
`data/reports/telecomts/<revision>/explainability/seed-<seed>/` contains the
global report, validation-derived feature and KPI tables, feature/KPI and
permutation charts, SHAP summary plots, and individual binary/diagnosis
explanation charts and JSON. These are descriptive reports, not model
performance claims.

## Existing application

- Start the FastAPI service: `python -m backend.main`
- Start Streamlit: `streamlit run frontend/app.py`
- Run the complete test suite: `python -m pytest`

The existing REST fields remain available and the request adds an optional
`kpi_names` list. When supplied, it must exactly match the canonical 18-channel
order; reordered or mislabeled named input is rejected. Inference CSV uploads
must carry those exact headers and order; the frontend rejects any mismatch
before sending values. A raw JSON matrix without channel labels remains
positional and is interpreted in canonical order because its order cannot be
inferred. The nine-page Streamlit dashboard sends inference requests only
through FastAPI.
Observed protocol strings are accepted only in the categorical channels;
categorical JSON nulls use the preprocessor's reserved `__MISSING__` category,
while missing numeric values are rejected.

`GET /health` reports API liveness separately from binary prediction readiness.
An HTTP 200 means the process is alive; check `prediction_ready` and
`readiness_status` for required detector, preprocessing, feature configuration,
and metadata compatibility. Optional diagnosis, attribution, and temporal
artifacts do not gate binary readiness and remain explicitly unavailable when
not configured. The Network Overview page displays the same liveness and
readiness states.

The shared inference pipeline validates the saved raw KPI/preprocessing
contract, metadata feature count/order, and fitted estimator feature names
before prediction. It reuses the saved preprocessor and the same feature
extractor as training. Binary detection uses the persisted threshold; only an
anomaly result proceeds to multiclass diagnosis. `/analyze` returns the full
diagnosis class-probability map when supported, along with per-stage model
versions. `/model-info` reports the saved threshold and pinned dataset revision.
The Streamlit diagnosis view displays the probability distribution when
available. Attribution and temporal localization remain optional; their
response fields stay null when unavailable rather than being fabricated.
Temporal localization receives the original 128-step sequence, separately
from the engineered feature vector.

The live artifacts were verified through the shared pipeline using one real
stored TelecomTS normal sample and one anomalous sample: the normal record
short-circuited diagnosis, and the anomalous record returned a diagnosis,
11-class probability map, attribution, and temporal localization. The
configured feature set has 250 ordered features, the saved detector and
diagnosis model orders match that configuration, and the saved threshold is
`0.429333...`. This is an artifact/wiring smoke check, not an evaluation result;
the reported evaluation metrics remain those produced by the pinned TelecomTS
evaluation workflows.

The latest application-level validation is documented in
[docs/integration-test-report.md](./docs/integration-test-report.md). It
includes the real-artifact HTTP flow, all nine page render checks, optional
adapter absence, and the Temporal Localization regression case.

## Reproducible installation and deployment

The verified runtime is Python `3.13.3` (recorded in `.python-version`).
`pyproject.toml` pins direct dependencies, `uv.lock` records the resolved
cross-platform dependency graph, and `requirements.txt` is an exported,
fully-versioned production dependency file. The broader
`requirements-test.txt` also locks the test and optional XGBoost training
dependencies. Install and verify locally from the repository root:

```powershell
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest -q
```

To intentionally update dependencies, install `uv==0.8.19`, update the pinned
requirements in `pyproject.toml`, then regenerate both lock representations:

```powershell
uv lock
uv export --frozen --no-dev --no-hashes --no-annotate --output-file requirements.txt
uv export --frozen --all-extras --no-hashes --no-annotate --output-file requirements-test.txt
```

Review the resolved changes and run the complete tests before deployment.
The `training` extra pins XGBoost; it is optional at inference and is excluded
from the production requirements to avoid installing an unnecessary training
dependency on hosted services.

### Separate FastAPI backend (Render)

`render.yaml` deploys the FastAPI service separately from the dashboard. It
binds to the host-provided `PORT`, requires production authentication and an
explicit CORS origin, and provisions a persistent `artifacts/` disk. The
blueprint uses a paid Starter service and persistent disk; review the hosting
provider's current charges before creating it.

Before the API can predict, securely provision the real generated files into
that mounted `artifacts/` directory. Required files are
`detection.joblib`, `preprocessing.joblib`, `feature_config.json`, and
`model_metadata.json`. Copy the optional diagnosis, attribution, and temporal
localization artifacts and their metadata/configuration files as well if those
features should be available. Generated artifacts are intentionally ignored by
Git and are not uploaded by this blueprint; an empty volume results in a live
but explicitly unready API. Do not put model artifacts or API keys in the Git
repository or public object storage.

When creating the Render service, provide these environment values through the
provider's secret/environment settings:

| Variable | Required value |
|---|---|
| `CL5G_API_KEY` | A randomly generated secret with at least 32 non-whitespace characters. |
| `CL5G_CORS_ORIGINS` | The exact HTTPS Streamlit app origin, without a path or trailing slash, e.g. `https://your-app.streamlit.app`. |
| `CL5G_ENV` | `production` (set in `render.yaml`). |
| `CL5G_API_HOST` | `0.0.0.0` (set in `render.yaml`). |
| `PORT` | Supplied by Render and consumed by the startup command. |

Use the provider's secret manager to generate/store the API key; never commit
it. Production startup fails closed if the key is weak/missing or CORS is
unset, wildcard, or malformed. `/health` is public for liveness/readiness
probes; `/predict`, `/analyze`, and `/model-info` require `X-API-Key`.
Authentication is shared-key protection, not user identity or per-user
authorization. Keep HTTPS enabled, rotate the key after exposure, and use
provider-side access controls/monitoring appropriate for a public research API.

After deployment, verify `https://<backend-host>/health` reports
`liveness_status: "alive"` and `prediction_ready: true`. Do not treat HTTP 200
alone as model readiness. `/docs` documents the request contract; protected
inference/model-info routes still require the API key.

### Streamlit Community Cloud frontend

Deploy `frontend/app.py` as the Streamlit app entry point using the root
`requirements.txt`; the dashboard does not host FastAPI or load model files.
Configure these values in the app's Streamlit Secrets settings:

```toml
CL5G_API_BASE_URL = "https://<backend-host>"
CL5G_API_KEY = "<same secret configured on the backend>"
```

The client uses the HTTPS API URL and sends the key in the `X-API-Key` header
for prediction, analysis, and model-info requests. Health remains public. Do
not use `localhost` for a separately hosted backend. The backend URL can also
be supplied with `CL5G_API_BASE_URL`; environment variables take precedence
over Streamlit secrets. If no deployment URL is configured, the local
development URL from `config.yaml` is used. When a key is configured, the
client sends it only to the exact configured backend URL; editing the dashboard
URL cannot redirect that secret to another host.

These deployment definitions prepare the services but do not deploy them or
provision the ignored model artifacts. Hosted-service startup, platform
accounts, secret entry, domain/CORS configuration, artifact transfer, and
live production traffic remain operator actions and have not been independently
verified from this workspace.

## Final research status

The evidence-based completion report is in
[docs/deployment-and-completion-report.md](./docs/deployment-and-completion-report.md).
It separates implemented functionality from test-double checks, real saved
artifact smoke checks, experimental evaluation, and functionality that cannot
be verified until external deployment. The project is not connected to a live
5G network and does not ingest streaming telemetry.

## Current validation results

Full ingestion of the pinned revision completed with:

| Measure | Count |
|---|---:|
| Loaded samples | 32,000 |
| Normal samples | 30,765 |
| Anomalous samples | 1,235 |
| Invalid samples | 0 |
| Source JSONL files | 33 |
| Samples with temporal annotations | 1,235 |
| Samples with affected-KPI annotations | 1,235 |

| Normalized label | Count |
|---|---:|
| Normal | 30,765 |
| Jamming | 279 |
| Co-Channel Interference (Mild) | 146 |
| Buffer Overflow (Gradual Buildup) | 138 |
| Co-Channel Interference (Severe) | 114 |
| Antenna Failure | 107 |
| Faulty RF Filters (Temporal) | 100 |
| High Network Congestion (Gradual Buildup) | 93 |
| Doppler Shift (Severe) | 86 |
| Faulty Handover Algorithm (Too Frequent) | 64 |
| Resource Allocation Bugs | 60 |
| High Network Congestion (Sudden Spike) | 48 |

The same counts, source revision/file list, per-file row counts, protocol
domains, and validation status are persisted in
`data/reports/telecomts/<revision>/validation.json`. The repeated-ingestion
test covers idempotent replacement rather than appending. Test fixtures
validate parser/error behavior; they are not used as TelecomTS data or
model-quality evidence.

Preprocessing and EDA outputs are separate from ingestion and contain no model
metrics. See the `eda/` reports and figures for measured dataset statistics
and data-quality results.

The full EDA run validated all 32,000 processed samples against their metadata:
zero invalid or duplicate records, zero missing/non-finite KPI values, and
zero ingestion errors. All 33 source files were represented. It found 1,235
inclusive temporal intervals; their mean duration was 91.81 steps and median
duration was 121 steps. The saved statistics report includes the per-KPI
numeric summaries and correlations. At the sequence-timestep level the
categorical counts were:

| Protocol KPI | Empty string | `"None"` | `"TCP"` | `"UDP"` |
|---|---:|---:|---:|---:|
| UL_Protocol | 15,374 | 1,365,949 | 2,662,170 | 52,507 |
| DL_Protocol | 15,465 | 1,491,990 | 2,535,290 | 53,255 |
