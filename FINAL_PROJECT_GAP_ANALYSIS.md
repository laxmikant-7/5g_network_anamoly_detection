# Final Project Gap Analysis

**Audit date:** 2026-11-22  
**Scope:** Read-only repository comparison against the nine-page dashboard and complete offline-training/online-inference requirements.  
**Status summary:** Core ingestion, preprocessing, source-aware splitting, feature extraction, binary/multiclass training, evaluation, attribution, experimental temporal localization, unified inference, API, and dashboard scaffolding are implemented and have genuine artifacts. Several dashboard/reporting gaps remain, and one legacy/stub implementation family is unrelated to the real TelecomTS temporal path.

## A. Executive summary

### Genuine implementation

- TelecomTS ingestion is implemented against immutable revision `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`.
- The local audit found 32,000 processed records in 33 source JSONL files, including 30,765 normal and 1,235 anomalous records, in `data/raw`, `data/processed`, and `data/reports`.
- Validation reports 0 invalid records and 0 missing/non-finite KPI values.
- Source-aware 70/10/20 train/validation/test partitioning is implemented. The split report has no source overlap.
- Stateless preprocessing uses 16 numeric channels plus one-hot encoding for `UL_Protocol` and `DL_Protocol`.
- Shared feature extraction produces the saved 250-feature `base_plus_temporal` contract.
- Binary Random Forest training/evaluation, multiclass Random Forest diagnosis training/evaluation, feature SHAP/permutation attribution, and supervised per-timestep temporal localization are implemented.
- Local genuine artifacts exist for detection, preprocessing, diagnosis, attribution, temporal localization, feature configuration, and metadata.
- FastAPI provides `/health`, `/predict`, `/analyze`, `/model-info`, validation, liveness/readiness separation, and API-key protection in production.
- Streamlit has exactly nine sidebar sections and uses the FastAPI API client for inference.
- Tests cover ingestion, preprocessing, EDA fixtures, splitting, feature extraction, binary and diagnosis training/evaluation logic, inference, attribution, temporal localization, API, CSV parsing, and dashboard rendering.

### Incomplete or not production-ready

- The dashboard does not by default load and display the genuine persisted binary/multiclass/temporal evaluation reports. Model Evaluation calculates uploaded actual/predicted CSV values instead.
- Network Overview does not load the genuine evaluation report; it only reads uploaded evaluation session state.
- Dataset Exploration lacks the full requested dataset-level box plots, correlation heatmap, and historical time-series controls, although it includes label distributions and histograms.
- KPI Monitoring has only a combined line chart and basic dimensions; it lacks individual KPI selection, statistical summaries, KPI comparison controls, and interval highlighting tied to localization output.
- Anomaly Detection does not display the saved classification threshold in its result UI.
- KPI Attribution displays API-ranked KPI tables and feature tables but not persisted global SHAP/permutation chart loading or an individual chart; it has no full visual global summary.
- Temporal localization is genuine and actual-annotation-derived, but explicit test evidence shows it does not beat the full-window test baseline. It must remain an experimental adapter.
- Two legacy modules remain stubs: `src/models/localization.py` and `src/evaluation/metrics.py`.
- The legacy `time_ordered` split function is explicitly unimplemented. This is a reserved/dead contract, not the active TelecomTS source-aware splitter.
- Multiclass diagnosis test performance is weak and the test partition is historical; this is genuinely measured, not complete operational readiness.
- Hosted deployment has not been tested. Artifact/model transfer and public Streamlit/FastAPI endpoints remain operator tasks.

## B. Nine-page dashboard audit

### 1. Network Overview

- **Existing:** Historical uploaded/available CSV records, record/entity/observed-category metrics, historical label distribution, liveness/readiness, loaded components, and supported anomaly categories in model info.
- **Missing:** Automatic loading of genuine persisted binary/multiclass/dataset reports; a separate supported-versus-observed category panel; validated model performance without a session upload; historical source totals from the configured processed TelecomTS store as a direct dataset rather than optional CSV discovery.
- **Backend dependencies:** `/health`, `/model-info`, optional `/analyze` state from prior session.
- **Data dependencies:** Top-level `*.csv` files under `data/raw` or `data/processed`, or an uploaded DataFrame.
- **Integration:** Implemented and no live prediction is confused with historical data.
- **Tests:** Dashboard AppTest render; test-only DataFrame; liveness/readiness tests.
- **Gap severity:** High for genuine persisted performance and source/support separation.

### 2. Dataset Exploration

- **Existing:** Upload `*.csv`; reads configured top-level raw/processed CSV files; normal/anomaly counts; observed category chart; numeric histograms with marginal boxes; normal-versus-anomalous KPI histogram; record preview.
- **Missing:** 128-step TelecomTS-specific sequence dataset loading, box plots for all numeric KPIs, correlation heatmap, historical time-series visualization, missing/categorical distribution integration, source-file distribution, temporal interval distribution, and robust validation against TelecomTS label/KPI contracts.
- **Backend dependencies:** None for exploration. Backend is only needed for separate inference pages.
- **Data dependencies:** Historical row-based CSV or uploaded CSV. The current loader does not recursively load TelecomTS JSONL or compressed sequences.
- **Integration:** Keeps exploration separate from individual prediction state.
- **Tests:** CSV parsing helpers and dashboard AppTest; no comprehensive nine-chart EDA dashboard test.
- **Gap severity:** High.

### 3. KPI Monitoring

- **Existing:** Upload 128×18 CSV/JSON; strict CSV header/order; JSON positional canonical order; combined Plotly line chart; source/step/channel metrics.
- **Missing:** Individual KPI selection, per-KPI statistical summary table, KPI comparison UI, missing-user-annotation highlighting, and localization interval shading.
- **Backend dependencies:** `/model-info` for channel-name resolution and `/analyze` state for a returned localization interval.
- **Data dependencies:** The shared uploaded sequence in Streamlit session state.
- **Integration:** Sequence and channel names are saved once and reused by diagnosis/localization.
- **Tests:** Sequence parser, dimensions, finite values, channel order, AppTest page rendering.
- **Gap severity:** Medium.

### 4. Anomaly Detection

- **Existing:** Strict upload and `/predict`; normal/anomaly status, anomaly probability, and model version.
- **Missing:** Explicit classification-threshold display and threshold-vs-probability explanation. The threshold is in `/model-info` but is not included in `_show_binary_result`.
- **Backend dependencies:** `/predict` and `/model-info`.
- **Data dependencies:** Validated 128×18 sequence.
- **Integration:** Correctly calls FastAPI and stores the shared sequence.
- **Tests:** API normal/anomaly, invalid inputs, named KPI order, and artifact readiness.
- **Gap severity:** Medium.

### 5. Anomaly Diagnosis

- **Existing:** Complete `/analyze`; normal short-circuit; predicted category; confidence; 11-class probability table; model version.
- **Missing:** A low-confidence/supported-class caveat based on actual model performance and clearer visual probability chart for all 11 classes.
- **Backend dependencies:** `/analyze`.
- **Data dependencies:** A binary-anomalous sequence and available diagnosis artifact.
- **Integration:** Diagnosis runs only after the binary result flags anomaly.
- **Tests:** Normal no-diagnosis and anomaly probability coverage; real-artifact smoke.
- **Gap severity:** Low-to-medium.

### 6. KPI Attribution

- **Existing:** API attribution tables for binary and diagnosis; top feature/KPI contributions; signed and absolute values; association-not-causality caption.
- **Missing:** Persisted global feature importance loading, permutation-importance chart loading, and SHAP summary chart loading. Individual model explanation charting is not exposed; only structured tables are rendered.
- **Backend dependencies:** `/analyze` with optional attribution artifact.
- **Data dependencies:** Saved attribution adapter and validation-derived reports.
- **Integration:** Optional and null-safe.
- **Tests:** Attribution model/unit tests, inference, API response, and dashboard rendering.
- **Gap severity:** Medium for the dashboard requirement, low for backend explainability.

### 7. Temporal Localization

- **Existing:** Original sequence pass-through, API interval, time-step/time-second mapping, experimental warning, score calibration status, and Plotly interval highlight.
- **Missing:** Affected-KPI prediction is not supported; ground-truth upload/comparison UI is absent. Annotation coverage and limitations are not shown as a complete per-page report.
- **Backend dependencies:** `/analyze` and temporal localization artifact.
- **Data dependencies:** Original 128×18 sequence and server-side TelecomTS annotations for evaluation.
- **Integration:** Implemented but correctly marked experimental.
- **Tests:** Interval conversion, context alignment, inclusive boundaries, time mapping, missing intervals, group overlap, metric plots, and populated temporal AppTest state.
- **Gap severity:** Medium. The core defect is model performance, not page wiring.

### 8. Model Evaluation

- **Existing:** User upload of actual/predicted CSV, strict ground-truth/predicted separation, accuracy, precision, recall, F1, confusion matrix, classification report, and binary precision-recall curve.
- **Missing:** F2-score, balanced accuracy, persisted genuine binary/diagnosis report loading, anomaly-class support, test-pristine/historical status, ROC loading, and automatic report directory selection.
- **Backend dependencies:** None. This is intentionally ground-truth evaluation.
- **Data dependencies:** Uploaded labeled/prediction CSV or future persisted report loader.
- **Integration:** Correctly does not calculate live-inference performance.
- **Tests:** Evaluation helper tests for uploaded labels and invalid/reused columns.
- **Gap severity:** High.

### 9. About the Project

- **Existing:** Project purpose, data-integrity caveats, workflow, model features, and no-live-network disclaimer.
- **Missing:** A complete data/model/evaluation limitations page: observed vs supported categories, model-specific metrics, TelecomTS data revision, temporal performance caveat, deployment status, algorithms, and architecture.
- **Backend dependencies:** Optional `/model-info`.
- **Data dependencies:** README and reports; currently mostly static/contract-derived copy.
- **Integration:** Implemented.
- **Tests:** Dashboard rendering test.
- **Gap severity:** Medium.

## C. ML pipeline audit

| Stage | Implementation status | Evidence | Remaining gap |
|---|---|---|---|
| Ingestion | Implemented | `src/data/telecomts.py`, 33 compressed source files, 32,000 processed records, validation JSON | Add a final independent dataset-card/schema snapshot test if revision changes. |
| Schema validation | Implemented | Exact 128×18 shape, 16 numeric/2 categorical channels, labels/interval/affected-KPI validation | Keep pinned revision validation and invalid-record audit for future revision updates. |
| Preprocessing | Implemented | Stateless fixed one-hot protocol encoding in `telecomts_preprocessing.py` | No learned transform is expected; preserve reject-not-impute behavior. |
| Feature engineering | Implemented | Shared 250-feature base+temporal extractor | Feature experiment JSON exists; no automatic comparison to alternative final feature sets is required for this implementation. |
| Group-aware split | Implemented | `src/data/split.py`, `split_metadata.json`, no source overlap | Legacy time-ordered splitter intentionally remains unimplemented. |
| Binary training | Implemented | `src/pipeline/train.py`, Random Forest selection and threshold tuning | Retraining remains guarded by one-time test state. |
| Binary evaluation | Implemented | Genuine held-out metrics and plots in `data/reports/.../binary_detection/seed-42` | Dashboard/report loader missing. |
| Diagnosis training | Implemented | `src/pipeline/train_diagnosis.py`, balanced Random Forest artifact | No fresh-pristine test remains because binary already used this test partition. |
| Diagnosis evaluation | Implemented with weak results | Genuine 339-anomaly historical holdout report | Improve data/model performance before operational claim. |
| Attribution | Implemented | Validation-only SHAP/permutation reports and `attribution.joblib` | Dashboard/global chart loading incomplete. |
| Temporal localization | Implemented as experimental | Annotation-derived model and metrics | Test mean IoU 0.7606 vs 0.7664 baseline; affected-KPI prediction absent. |
| Unified inference | Implemented | `src/pipeline/inference.py` shared with FastAPI | Normal short-circuit, binary threshold, diagnosis, attribution, and localization are integrated. |
| Deployment | Prepared only | `render.yaml`, env settings, lockfiles | No hosted service/artifact transfer/public traffic evidence. |

### Key genuine metrics

#### Binary detector

- Test samples: 6,352; normal 6,013; anomalous 339.
- Accuracy: 0.968986.
- Anomaly precision: 0.706395.
- Anomaly recall: 0.716814.
- F1: 0.711567.
- F2: 0.714706.
- Average precision: 0.782476.
- ROC-AUC: 0.950008.
- False-positive rate: 0.016797.
- Confusion matrix: `[[5912, 101], [96, 243]]`.
- The test set was initially pristine for binary evaluation.

#### Multiclass diagnosis

- Historical anomaly-only test samples: 339.
- Accuracy: 0.457227.
- Balanced accuracy: 0.262586.
- Macro precision: 0.379890.
- Macro recall: 0.262586.
- Macro F1: 0.232568.
- Weighted F1: 0.358676.
- All 11 classes had test support. Jamming recall was 1.0, Antenna Failure 0.818182, but several classes had zero recall. The test partition is explicitly historical because binary evaluation preceded diagnosis evaluation.

#### Temporal localization

- 1,235 valid anomalous annotations; 30,765 valid normal empty annotations.
- 593 full-sequence anomaly annotations; median interval 121 steps.
- Selected threshold: 0.42.
- Validation mean temporal IoU: 0.814436.
- Test mean temporal IoU: 0.760567.
- Full-window test baseline mean temporal IoU: 0.766409.
- Therefore, supervised test performance did not exceed the trivial full-window baseline.

## D. Architecture comparison

### Intended offline training architecture

TelecomTS ingestion → validation → preprocessing → source-grouped split → shared feature extraction → binary and diagnosis candidate fitting → validation/model/threshold selection → held-out evaluation → serialization.

The implementation follows this architecture. Training and inference share `TelecomTSPreprocessor` and `extract_features`. Labels and annotations are excluded from predictive inputs.

### Intended online inference architecture

Upload/CSV/JSON input → Pydantic validation → preprocessing → shared feature extraction → saved binary detector → threshold → optional diagnosis → optional attribution → optional temporal localization → structured API response.

The implementation follows this architecture. Normal detections short-circuit diagnosis, attribution, and localization. A response explicitly uses nulls for unavailable optional modules.

### Existing implemented architecture

Streamlit is a presentation client, FastAPI is the API boundary, and `InferencePipeline` owns preprocessing/artifact loading/prediction. This separation is intact and should not be replaced.

The remaining architectural risks are:

- Persisted report loading is absent from the dashboard.
- Dashboard evaluation is still upload-only rather than report-aware.
- Legacy localization interfaces and metrics do not match the actual temporal annotation implementation and are not used by unified inference.
- Deployment configuration exists, but hosted artifact provisioning is external and unverified.

## E. Data contract

### Canonical raw KPI order

The exact order in `src/data/telecomts.py` and `artifacts/feature_config.json` is:

1. RSRP
2. DL_BLER
3. DL_MCS
4. UL_BLER
5. UL_MCS
6. UL_NPRB
7. UL_SNR
8. TX_Bytes
9. RX_Bytes
10. Estimated_UL_Buffer
11. PRBs_DL_Current
12. PRBs_UL_Current
13. PRB_Utilization_DL
14. PRB_Utilization_UL
15. UL_Protocol
16. UL_NumberOfPackets
17. DL_Protocol
18. DL_NumberOfPackets

### Numerical/categorical split

- **Numerical:** 16 KPI channels, from RSRP through the numerical packet/PRB channels listed in `NUMERIC_KPIS`.
- **Categorical:** `UL_Protocol`, `DL_Protocol`.
- **Observed protocol values:** `""`, `"None"`, `"TCP"`, `"UDP"`.
- **Reserved transform value:** `"__MISSING__"` for JSON null during preprocessing.
- The empty string and literal `"None"` remain distinct.

### CSV input

- Exactly 128 rows and 18 columns.
- Header names exactly equal the canonical names and order.
- Numeric channels must be finite numeric values.
- Categorical channels contain supported protocol strings or null-like values.
- Null categorical values are accepted by the API where categorical channels are concerned.

### JSON input

- Array shape 128×18 or a JSON object containing `"sample": 128×18`.
- JSON arrays have no channel names and are interpreted in canonical positional order.
- Protocol channels may contain `""`, `"None"`, `"TCP"`, `"UDP"`, or null.
- Numeric channels must contain finite numbers, not strings or null.

### Internal model input

- Raw 128×18 sequence.
- Stateless preprocessing creates 128×26 values: 16 numeric channels plus five one-hot columns for each protocol KPI.
- Shared `extract_features` emits 250 ordered features for `base_plus_temporal`.
- No label, description, interval, affected-KPI, source, zone, application, mobility, congestion, timestamp, or sample metadata enters the model input.

## F. Model artifact requirements

The unified inference pipeline requires:

| Artifact | Purpose | Required |
|---|---|---|
| `artifacts/detection.joblib` | Binary detector | Yes |
| `artifacts/preprocessing.joblib` | Stateless TelecomTS sequence encoder | Yes |
| `artifacts/feature_config.json` | Raw/processed KPI names and ordered 250-feature contract | Yes |
| `artifacts/model_metadata.json` | Model version, threshold, dataset revision, feature count/order | Yes |
| `artifacts/diagnosis.joblib` | 11-class anomaly diagnosis | Optional for binary-only `/predict`; required for complete `/analyze` anomaly response |
| `artifacts/diagnosis_class_mapping.json` | Diagnosis class mapping | Used/retained with diagnosis metadata |
| `artifacts/diagnosis_metadata.json` | Diagnosis version and class metadata | Optional/additional diagnosis contract |
| `artifacts/diagnosis_configuration.json` | Diagnosis estimator configuration | Optional/reporting configuration |
| `artifacts/attribution.joblib` | Genuine local KPI/feature explanations | Optional |
| `artifacts/temporal_localization.joblib` | Experimental supervised interval localizer | Optional |

Additional compatibility checks include:

- Feature JSON input KPI names equal canonical KPI order.
- Metadata feature order equals feature JSON.
- Detector and diagnosis estimators expose and match saved feature names.
- Preprocessor exposes `transform`.
- Model metadata threshold is a finite value in `[0,1]`.
- Optional artifacts remain null instead of producing fabricated output.

## G. Prioritized implementation roadmap

### Phase 1: Persisted genuine evaluation integration

- **Purpose:** Make Network Overview and Model Evaluation display genuine persisted reports rather than only uploaded labels.
- **Expected input:** Binary, multiclass, split, and temporal report JSON/CSV/PNG paths.
- **Expected output:** Verified report cards, confusion matrices, PR/ROC curves, supported-class metrics, test-pristine status.
- **Files to create:** `frontend/evaluation_reports.py`; optional backend `GET /evaluation-report` or a dashboard-local report loader.
- **Files to modify:** `frontend/app.py`, `frontend/dashboard_data.py`, `backend/main.py`/schemas if API loading is chosen.
- **Tests:** Report loader, missing/corrupt report handling, test-history labels, no live-label metric synthesis.
- **Dependencies:** Existing report JSON, split metadata, binary/multiclass/temporal report paths.
- **Completion:** All genuine report values are loaded by explicit artifact/revision/seed and no unverified metrics are shown.

### Phase 2: Complete KPI Monitoring

- **Expected input:** Shared 128×18 upload and API localization interval.
- **Expected output:** Per-KPI Plotly chart, individual KPI selector, summary table, comparison, and highlighted interval.
- **Files to create:** Monitoring helper module if page logic grows.
- **Files to modify:** `frontend/app.py`, `frontend/dashboard_data.py`, tests.
- **Tests:** KPI selection, numeric/categorical handling, interval bounds, 128-step alignment, unavailable interval state.
- **Dependencies:** `/analyze` localization result and canonical channel names.
- **Completion:** Each selected KPI has summary/time-series/comparison and a visible interval when real server output exists.

### Phase 3: Complete historical Dataset Exploration

- **Expected input:** Uploaded/TelecomTS historical sequence and metadata; optional labeled row CSV.
- **Expected output:** Summary, category distribution, histograms, box plots, correlation, time series, source distribution, temporal interval/missing/categorical distributions.
- **Files to create:** `src/data/dashboard_dataset.py` or `frontend/dataset_exploration.py`.
- **Files to modify:** `frontend/app.py`, `frontend/dashboard_data.py`.
- **Tests:** Recursive source discovery, JSONL/CSV parser, missing labels, 128-step shape, non-mixing with inference.
- **Dependencies:** Processed sequences/metadata and ingestion report.
- **Completion:** Every dataset visualization is backed by actual selected records, and synthetic fixture counts are clearly labeled.

### Phase 4: Complete Model Evaluation metrics

- **Expected input:** Ground truth/predicted label CSV and optional probability column.
- **Expected output:** Accuracy, precision, recall, F1, F2, balanced accuracy, multiclass support, confusion matrix, PR curve, and classification report.
- **Files to create:** `src/evaluation/uploaded.py` only if shared outside Streamlit.
- **Files to modify:** `frontend/dashboard_data.py`, `frontend/app.py`, tests.
- **Tests:** Binary/multiclass F2, balanced accuracy, missing support, one-class/zero-support, invalid probability.
- **Dependencies:** Uploaded actual/predicted labels and the current label taxonomy.
- **Completion:** No live predictions contribute to metrics; all results use explicit ground truth.

### Phase 5: Attribution dashboard enhancement

- **Expected input:** Saved global/permutation CSV/PNG/JSON and API local result.
- **Expected output:** SHAP summary, global/permutation charts, local feature/KPI chart, exact aggregation caveat.
- **Files to create:** `frontend/attribution_views.py` or a report loader.
- **Files to modify:** `frontend/app.py`, tests.
- **Tests:** Missing reports, chart fallback, binary/multiclass local values, association-not-causality text.
- **Dependencies:** Existing explainability artifacts/reports.
- **Completion:** Dashboard exposes report-backed global and local explanations without fabricating scores.

### Phase 6: Temporal localization quality and ground-truth comparison

- **Expected input:** A real 128×18 sequence plus server interval and optional uploaded ground-truth interval.
- **Expected output:** Group-aware metrics, boundary comparison, affected-KPI model output only if learned, and clear experimental status.
- **Files to create:** Trained affected-KPI module only if feasible.
- **Files to modify:** Temporal localizer, API schema, frontend page, report generator.
- **Tests:** IoU, event metrics, boundary MAE, timestep metrics, source overlap, missing affected labels.
- **Dependencies:** Valid annotation reliability and fresh held-out test.
- **Completion:** Reported model statistically supports improvement over baseline; if not, keep explicitly experimental.

### Phase 7: Diagnosis improvement

- **Expected input:** Genuine anomalous source-aware train/validation/test records.
- **Expected output:** Improved supported-class macro/weighted metrics and a pristine test result if available.
- **Files to create:** None required before a new experiment design.
- **Files to modify:** Training candidate/feature configuration, evaluation report, README.
- **Tests:** Class support, no source overlap, no synthetic resampling leakage, per-class reports.
- **Dependencies:** Source-aware partitions and model-selection policy.
- **Completion:** Genuine, reproducible validation improvement and a new untouched test evaluation.

### Phase 8: Residual cleanup and deployment verification

- **Expected input:** Deployment manifests, secrets, persisted artifacts, public URLs.
- **Expected output:** Live health/readiness, HTTPS API, correct Streamlit backend URL/key, and artifact-backed predictions.
- **Files to create:** Hosted smoke-check script/report template if needed.
- **Files to modify:** Deployment docs and environment validation only after actual provider testing.
- **Tests:** Production startup, public health, unauthorized/protected routes, real-artifact inference, Streamlit end-to-end.
- **Dependencies:** Provider account, persistent volume, generated artifacts, secrets, CORS origin.
- **Completion:** Only declare deployment-ready after independent public checks and artifact inventory verification.

## H. Known defects

### Temporal Localization page bug

The previously reported `UnboundLocalError` is not present in the audited current implementation. `_render_temporal_localization` initializes `values, sequence_names`, returns if no sequence, then assigns `names = _resolve_channel_names(...)`; `interval` is read after `analysis` validation. A regression test exists and a populated-sequence/analysis AppTest case was reported in prior integration work. It should remain in the test suite.

### Additional confirmed defects/gaps

1. **No persisted evaluation report loader:** Overview and Model Evaluation do not load actual binary/multiclass report JSON automatically.
2. **Detection threshold omitted from UI:** The detection result helper displays probability/status/model but not classification threshold.
3. **Incomplete Dataset Exploration:** Missing all-row correlation, general box plots, and historical sequence/time-series controls.
4. **Incomplete KPI Monitoring:** No individual channel selector/statistics/comparison/interval highlight despite partial sequence plotting.
5. **Incomplete KPI Attribution dashboard:** Global/permutation/individual visualizations are on disk but not loaded in Streamlit.
6. **Legacy stubs remain:** `src/models/localization.py`, `src/evaluation/metrics.py`, and `split_time_ordered` are real `NotImplementedError` code.
7. **Config/implementation separation:** `config.yaml` includes generic cross-layer fields and a disabled `localization` task, while active implementation is TelecomTS 18-KPI temporal interval localization. The names are stale but do not affect the active path.
8. **Test partition historical:** Diagnosis and localization do not use an untouched test set because binary evaluation already used it.
9. **Weak diagnosis:** Many anomaly classes have low or zero held-out recall.
10. **Experimental localization:** It did not beat the test baseline.
11. **Hosting unverified:** Render/Streamlit setup is prepared but not live-verified.
12. **`.venv` and caches are workspace artifacts:** Do not treat generated environment or cache directories as source implementation.

## I. Final acceptance criteria

The project is complete only when all of the following are true:

1. Exactly nine named pages remain available and no extra primary navigation section is introduced.
2. Real pinned TelecomTS data ingests to 32,000 records with 33 source files and zero invalid records.
3. Processed sequences preserve 128 steps and canonical 18-channel order; 16 numerical and 2 categorical channels are validated.
4. Labels and temporal/affected-KPI annotations match the pinned source and are preserved outside predictive inputs.
5. Train/validation/test partitions are source-file-disjoint and preserve every sample exactly once.
6. Training and inference use the same preprocessor and 250-feature extractor with identical order.
7. The binary artifact and decision threshold are genuine, versioned, loadable, and display their threshold in the dashboard.
8. Diagnosis supports only the 11 original classes and reports missing supported-class metrics accurately.
9. Binary/diagnosis test claims come only from saved report files and include the test-history caveat.
10. Attribution output comes only from the SHAP/permutation artifact and clearly states associations are not causality.
11. Temporal localization uses real annotations, reports real group-aware metrics, and remains experimental unless a held-out test demonstrably improves over a documented baseline.
12. Dataset Exploration supports uploaded TelecomTS/history CSV or processed data and displays genuine distributions, box plots, correlations, and sequences.
13. KPI Monitoring supports individual KPI selection, summaries/comparison, and real interval highlighting.
14. Model Evaluation supports F2 and balanced accuracy and loads actual persisted reports or computes only from explicit actual/predicted ground truth.
15. `/health` distinguishes liveness from readiness; protected inference routes reject missing keys; optional modules do not become fabricated output.
16. All actual model artifacts are provisioned and artifact/report checks pass on a hosted backend.
17. Streamlit reaches FastAPI through an environment/secret URL and passes real end-to-end normal/anomalous sequences.
18. Unit, integration, dashboard, real-artifact, and deployment smoke tests pass and test-double evidence is separated from genuine results.
19. README and completion reports accurately state implementation, experimental validation, unverified deployment, and limitations.
20. No live 5G integration or real-time telemetry claim is made unless real telemetry capture, streaming, and monitoring are separately implemented and tested.
