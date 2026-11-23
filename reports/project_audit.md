# Project Audit

**Audit date:** 2026-11-22  
**Repository:** Cross-Layer 5G Network Anomaly Localization and Diagnosis  
**Dataset revision:** `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`  
**Scope:** Existing source, API, nine Streamlit sections, configuration, local model artifacts, tests, training/evaluation outputs, and the dashboard historical CSV.

## Executive summary

This is an established research application, not an empty scaffold. It has a pinned TelecomTS ingestion pipeline, validation and EDA, source-file-aware partitions, shared preprocessing and feature extraction, trained detection and diagnosis artifacts, attribution and experimental temporal localization, unified inference, a FastAPI service, and a nine-section Streamlit dashboard. These components should be preserved.

The local dashboard CSV and its provenance were independently checked. The export contains 134,528 rows for 1,051 stratified sample entities, with 12 observed labels. Its class proportions do **not** represent the source dataset prevalence. The original processed metadata contains 32,000 samples: 30,765 normal and 1,235 anomalous. The dashboard CSV is for historical exploration and should not replace the complete sequence set for training.

The important remaining gaps are dashboard reporting and research quality, not missing end-to-end plumbing. The dashboard does not automatically load the genuine saved evaluation reports. Dataset Exploration and KPI Monitoring are less comprehensive than the requested visual analysis. The multiclass model has weak test performance; the temporal model does not outperform a full-window baseline. Hosting is prepared but not verified live.

No existing application or source dataset was changed for this audit. The new CSV quality-audit utility writes separate reports and figures under `reports/`.

## Repository structure

| Path | Purpose and observed contents |
|---|---|
| `src/common/` | Shared project errors, logging, and project paths. |
| `src/data/` | TelecomTS ingestion/schema, EDA, preprocessing, grouped data splitting, and the new historical-CSV quality auditor. |
| `src/features/` | Ordered, shared feature configuration and extraction used by training and inference. |
| `src/models/` | Binary detector, diagnosis classifier, KPI attribution adapter, temporal localizer, and legacy layer-localization interfaces. |
| `src/evaluation/` | Binary and multiclass metrics, attribution, and temporal-localization evaluation/reporting. |
| `src/pipeline/` | Configuration, training, diagnosis training, inference, and capability reporting. |
| `backend/` | FastAPI app, routes, dependencies, and Pydantic request/response contracts. |
| `frontend/` | Streamlit app with nine sections, API client, shared UI components, and CSV/sequence helpers. |
| `artifacts/` | Local joblib models, feature configuration, and model metadata. Model artifacts are local and are not assumed to be provisioned by hosted deployments. |
| `data/raw/`, `data/processed/`, `data/reports/` | Downloaded source archives, processed records/splits, and revision-specific ingestion/training/EDA evidence. |
| `tests/` | Unit, integration, API, data, model, and dashboard tests. |
| `docs/`, `notebooks/` | Deployment/integration notes and research exploration. |
| `config.yaml`, `pyproject.toml`, `requirements*.txt`, `uv.lock`, `.python-version` | Runtime paths/schema, pinned dependencies, lock, and Python version. |
| `render.yaml` | Backend hosting blueprint; a live deployment was not verified. |

The Python inventory contains the application modules, dashboard/API modules, generator, and tests; individual source modules and declared functions/classes were enumerated. Existing README and gap/completion reports were also cross-checked against the current files and local outputs.

## Implemented functionality

### Data and ML pipeline

- [`../src/data/telecomts.py`](../src/data/telecomts.py) reads the pinned TelecomTS source archives, validates the original record schema and annotations, preserves source IDs and stable sample IDs, and writes processed sequences and metadata separately.
- [`../src/data/telecomts_preprocessing.py`](../src/data/telecomts_preprocessing.py) validates canonical input order and encodes the two protocol channels without treating them as numeric values. Predictive input is limited to the KPI sequence.
- [`../src/data/split.py`](../src/data/split.py) implements source-file-grouped training, validation, and test partitions. The saved split report records no cross-partition source overlap.
- [`../src/features/extraction.py`](../src/features/extraction.py) and [`../src/features/builder.py`](../src/features/builder.py) define ordered engineered features shared with inference.
- [`../src/pipeline/train.py`](../src/pipeline/train.py) and [`../src/pipeline/train_diagnosis.py`](../src/pipeline/train_diagnosis.py) train binary and multiclass estimators using saved partitions and validation-based selection. The final test-evaluation state is recorded to prevent accidental repeated use from being represented as a fresh test.
- [`../src/evaluation/binary.py`](../src/evaluation/binary.py), [`../src/evaluation/multiclass.py`](../src/evaluation/multiclass.py), and [`../src/evaluation/temporal_localization.py`](../src/evaluation/temporal_localization.py) generate real evaluation artifacts.
- [`../src/models/attribution.py`](../src/models/attribution.py) maps feature-level explanation values back to KPI channels. Attribution is an association with model output, not causation.
- [`../src/models/temporal_localization.py`](../src/models/temporal_localization.py) consumes the original sequence. The implementation and dashboard label it experimental; it does not claim network-layer localization.

### API and dashboard

- [`../backend/main.py`](../backend/main.py) exposes `/health`, `/predict`, `/analyze`, and `/model-info`. Health distinguishes process liveness from prediction readiness; production configuration requires an API key and explicit HTTPS CORS origins.
- [`../backend/schemas.py`](../backend/schemas.py) validates the 128 × 18 request, finite numerical values, categorical protocol domain, and optional explicit KPI order.
- [`../src/pipeline/inference.py`](../src/pipeline/inference.py) owns artifact loading, preprocessing, features, binary thresholding, conditional diagnosis, and optional attribution/localization. FastAPI does not implement a separate predictor.
- [`../frontend/app.py`](../frontend/app.py) contains nine Streamlit sections. Inference is sent through [`../frontend/api_client.py`](../frontend/api_client.py); CSV parsing and dashboard data helpers are in [`../frontend/dashboard_data.py`](../frontend/dashboard_data.py).
- The previously reported Temporal Localization `names` assignment defect is fixed: channel names are resolved before rendering the selected KPI and interval. The test `test_temporal_localization_page_with_sequence_and_analysis_is_stable` covers a populated analysis and sequence.

## Nine-section dashboard audit

| Section | Existing functionality | Remaining gap | Dependencies / tests |
|---|---|---|---|
| **Network Overview** | Historical record/entity/observed-category counts, class distribution, service liveness/readiness, loaded components, configured supported categories. | Does not automatically read genuine persisted model evaluation reports; observed and configured categories/performance are not all surfaced as distinct data-backed panels. | Historical CSV loader, `/health`, `/model-info`; AppTest and API readiness coverage. |
| **Dataset Exploration** | CSV upload and configured top-level CSV discovery; label distribution; numeric selection/histograms; normal/anomaly comparison; record preview. | Missing integrated full dataset EDA panels such as correlation heatmap, broad box-plot views, historical time-series controls, and TelecomTS-specific validation presentation. The separate audit now generates requested static quality figures. | Pandas/Plotly; dashboard helper and render tests, plus the new auditor tests. |
| **KPI Monitoring** | Strict 128 × 18 sequence upload; CSV headers/order and JSON dimensions validated; sequence plotted and shared in session state. | Current UI emphasizes a combined trace; per-KPI descriptive summaries/comparison controls remain limited. | `dashboard_data.parse_kpi_sequence`, `/model-info`; CSV/JSON parser and render tests. |
| **Anomaly Detection** | Validated sequence submitted to `/predict`; shows binary result/probability and model version. | Decision threshold is available from model info but is not fully explained in the result presentation. | FastAPI `/predict`; API and dashboard tests. |
| **Anomaly Diagnosis** | Shared `/analyze` request, conditional diagnosis, category probabilities/confidence, and unavailable states. | Performance caveat could be more prominent alongside prediction confidence. | Diagnosis artifact and shared inference; real-artifact smoke and API tests. |
| **KPI Attribution** | Structured TreeSHAP feature/KPI contributions shown in tables with association-not-causation language. | Persisted global explanation plots and a more visual per-sample contribution chart are not fully surfaced. | Optional attribution artifact and `/analyze`; attribution/inference/API/UI tests. |
| **Temporal Localization** | Original sequence retained; interval and score visualized; experimental status shown; no unsupported interval is fabricated. | Affected-KPI prediction is unavailable; ground-truth comparison is not an interactive dashboard workflow. Test performance does not beat the full-window baseline. | Optional temporal artifact and annotations; temporal and populated-page regression tests. |
| **Model Evaluation** | Upload actual/predicted labels; confusion matrix, classification report, and binary PR curve are computed only from labeled evaluation data. | Uses uploaded results rather than loading the saved genuine held-out reports; F2/balanced accuracy and persisted ROC/PR reports are not fully exposed. | Evaluation helper; ground-truth-only tests. |
| **About the Project** | Research purpose, workflow, limitations, and no-live-telemetry caveats. | Should surface current evidence and weak model/localization limitations more explicitly and link to persisted reports. | Static app content; dashboard rendering test. |

## API contract and online architecture

The existing online flow is:

`SampleRequest (128 × 18) → strict schema/order validation → saved preprocessing → shared feature extraction → saved binary detector and threshold → conditional diagnosis → optional attribution/localization → structured Pydantic response`

`/predict` returns binary status, anomaly probability, and detector version. `/analyze` additionally returns diagnosis, probabilities, optional attributions/localization, and model metadata. Normal samples short-circuit diagnosis. Missing optional modules remain unavailable rather than being replaced with guessed outputs. The `/health` HTTP status indicates liveness; `prediction_ready` indicates whether required artifacts loaded.

There are no duplicate inference implementations in the Streamlit or FastAPI layers. The online API does not train models.

## Dataset and feature contract

The historical CSV schema is `timestamp`, `entity_id`, `layer`, `anomaly_type`, followed by the canonical 18 KPI channels:

`RSRP, DL_BLER, DL_MCS, UL_BLER, UL_MCS, UL_NPRB, UL_SNR, TX_Bytes, RX_Bytes, Estimated_UL_Buffer, PRBs_DL_Current, PRBs_UL_Current, PRB_Utilization_DL, PRB_Utilization_UL, UL_Protocol, UL_NumberOfPackets, DL_Protocol, DL_NumberOfPackets`.

There are 16 numerical KPI channels and two categorical protocol channels. `""`, `"None"`, `"TCP"`, and `"UDP"` are distinct observed categorical values. Each sample has 128 values per KPI. The CSV's `anomaly_type` is a target label; `entity_id` and `timestamp` are grouping/time fields; `layer=ran_du` is an assigned compatibility value and **not** original TelecomTS layer ground truth. The trained feature pipeline uses KPI data only and must keep the entire entity together in a split.

### Independently verified historical CSV

The read-only quality audit is in [`dataset_quality_report.md`](dataset_quality_report.md) and [`dataset_quality_report.json`](dataset_quality_report.json); figures are under [`figures/`](figures/).

- 134,528 rows × 22 columns; 1,051 entities; exactly 128 rows per entity.
- 12 observed labels: 100 Normal and 951 anomalous samples; row totals are 12,800 Normal and 121,728 anomalous.
- Canonical header order and KPI names match exactly.
- 0 missing/nonnumeric/non-finite numeric values; protocol values are in the permitted domain.
- 0 invalid timestamps, duplicate complete rows, duplicate entity/timestamp pairs, label inconsistencies, or schedule mismatches.
- Every selected entity matched original processed metadata; all 32,000 metadata labels matched the source-to-taxonomy mapping.
- File size is 39,232,764 bytes (37.415 MiB), below the dashboard's 200 MB cap.
- Tukey 1.5×IQR outliers were reported descriptively, not removed or called invalid.

The stratified CSV has 9.51 anomaly samples per normal sample, unlike the complete dataset (1,235 anomalous versus 30,765 normal). Do not use the historical export as a prevalence-representative training or evaluation set.

## Model artifact inventory and genuine evaluation

Local artifacts exist for `detection.joblib`, `diagnosis.joblib`, `preprocessing.joblib`, `feature_config.json`, `model_metadata.json`, `diagnosis_class_mapping.json`, diagnosis metadata/configuration, `attribution.joblib`, and `temporal_localization.joblib`. They are local files, not automatically bundled with a hosted FastAPI service. A real-artifact TestClient smoke against actual held-out TelecomTS sequences reported prediction readiness and returned a normal binary result plus an anomalous analysis with diagnosis, attribution, and temporal localization.

Saved evaluation evidence was read from the revision-specific reports:

- **Binary detection:** 6,352-sample test partition; accuracy 0.968986, anomaly precision 0.706395, recall 0.716814, F1 0.711567, F2 0.714706, average precision 0.782476, ROC-AUC 0.950008; confusion matrix `[[5912, 101], [96, 243]]`. This was the first test use recorded by its report.
- **Multiclass diagnosis:** 339 anomaly-only test samples; accuracy 0.457227, balanced accuracy 0.262586, macro-F1 0.232568, weighted-F1 0.358676. This partition is explicitly marked historical because the binary evaluation used it first. Multiple categories have very low or zero recall.
- **Temporal localization:** test mean IoU 0.760567 versus 0.766409 for the full-window baseline. It is experimental and did not outperform the baseline; no operational localization claim is justified.

These are saved experimental results, not current live-inference metrics.

## Configuration, dependencies, and tests

- Python version is pinned to 3.13.3; [`../pyproject.toml`](../pyproject.toml) and [`../uv.lock`](../uv.lock) pin the environment. `requirements.txt` and `requirements-test.txt` are present.
- [`../config.yaml`](../config.yaml) centralizes data paths, schema/taxonomy, split options, inference artifact paths, API settings, and logging.
- [`../render.yaml`](../render.yaml) prepares a separate backend deployment; no hosted service was tested.
- Existing tests cover ingestion/schema, preprocessing, splitting, features, training/evaluation behavior, attribution, temporal localization, API validation/readiness, frontend parsing/rendering, and generator behavior.
- Full local test command after adding the quality auditor: `.\.venv\Scripts\python.exe -m pytest -o addopts= -q` — **150 passed**.
- Genuine-artifact API smoke used actual held-out sequences: `/health` was prediction-ready; the selected normal sequence returned `normal`; the selected Jamming test sequence returned `anomaly` and included diagnosis, attribution, and temporal results.

## Known incomplete or legacy code

- [`../src/evaluation/metrics.py`](../src/evaluation/metrics.py) still raises `NotImplementedError` for legacy localization/diagnosis metric functions; active TelecomTS evaluation uses the concrete evaluation modules described above.
- [`../src/models/localization.py`](../src/models/localization.py) contains unimplemented legacy layer-prediction interfaces. No trained layer-localization artifact exists, and the dataset's assigned `ran_du` value is not evidence for cross-layer localization.
- `split_time_ordered` in [`../src/data/split.py`](../src/data/split.py) is explicitly unimplemented; the configured and used method is source-file-grouped splitting.
- Deployment configuration is not evidence of hosted artifact provisioning, public endpoint security, or production traffic.
- Diagnosis and temporal localization have material research limitations described above. No live 5G telemetry or real-time monitoring is integrated.

## Prioritized implementation roadmap

1. **Surface genuine evaluation artifacts in the dashboard.** Add a report loader for the saved binary, diagnosis, and temporal results; show test-set history/pristine status and separate supported categories from observed categories. Do not compute metrics from live inference.
2. **Complete Dataset Exploration and KPI Monitoring.** Add interactive heatmap/box/time-series exploration and per-KPI summaries, keeping the historical CSV separate from individual inference state.
3. **Strengthen research evaluation before changing models.** Establish a fresh, group-aware held-out split for future model selection/evaluation because the current final test data has already been used. Use the complete sequence dataset, not the stratified dashboard CSV.
4. **Investigate weak diagnosis and temporal results.** Select models only on validation data; compare localization with the full-window baseline and report results honestly. Do not claim cross-layer labels that the dataset does not contain.
5. **Verify hosted deployment separately.** Provision model artifacts securely, configure API secrets and dashboard URL, then test hosted liveness, readiness, authorization, CORS, and actual predictions.
6. **Keep legacy interfaces explicit.** Either retire/deprecate unused stubs after dependency checks or implement them only if genuine layer labels and a validated use case become available.

Do not rebuild the application or retrain solely to make the dashboard appear ready. Existing shared inference, artifacts, and data lineage are the foundation for future changes.
