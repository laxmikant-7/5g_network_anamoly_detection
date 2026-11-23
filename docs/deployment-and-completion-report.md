# Deployment and Project Completion Report

**Verification date:** 2026-11-22  
**Runtime:** Windows, Python 3.13.3  
**Dataset:** TelecomTS revision `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`  
**Automated tests:** 136 passed in the exact `uv.lock` environment

## Executive summary

The existing application and research pipeline are implemented and have
passed local automated and real-artifact checks. Reproducible dependency
locks, a separate FastAPI deployment blueprint, production authentication,
and Streamlit secret-based backend configuration are prepared.

This is not evidence of a hosted production deployment. Model files are
generated locally and ignored by Git; operators must securely provision them
on the backend artifact volume. Render/Streamlit account setup, secret entry,
public HTTPS checks, and production traffic remain unverified. The diagnosis
model's held-out performance is weak, and the temporal localizer is
experimental; these limitations are not hidden by the deployment readiness
work.

## Acceptance criteria and evidence

| Criterion | Status | Evidence and qualification |
|---|---|---|
| Real TelecomTS data ingested | Verified | Pinned source revision; 32,000 samples from 33 source JSONL files, 30,765 normal, 1,235 anomalous, zero invalid. The README documents category counts and data report locations. |
| Preprocessing works | Verified | Schema/data quality and protocol encoding are tested; the full EDA validation reports no invalid/duplicate records or missing/non-finite KPI values. |
| Features reproducible | Verified | A shared ordered 250-feature extractor is used by training and inference; parity, deterministic ordering, category encoding, and finite outputs are tested. |
| Group-aware evaluation | Verified with limitation | Train/validation/test splits have disjoint source-file groups and preserve all records. The test split has already been used for binary and diagnosis reporting, so it is historical, not a pristine untouched test set. |
| Trained binary detection | Verified | The saved Random Forest and threshold load and infer real held-out samples. The pinned report records anomaly precision 0.7064, recall 0.7168, F1 0.7116, and ROC-AUC 0.9500. |
| Multiclass diagnosis | Verified with limitation | The saved model returns an 11-class probability distribution for an anomalous real sequence. Held-out macro-F1 was 0.2326; several supported classes have zero recall. See the category-level table in the README. |
| KPI attribution | Available and smoke-tested | The saved TreeSHAP adapter returned binary and diagnosis attribution on the real-artifact analysis path. Importance describes model association, not physical causality. |
| Temporal localization | Experimental | The saved per-timestep model returns an interval and localization score. Test mean IoU was 0.7606 versus 0.7664 for the full-window baseline; it did not outperform the baseline and is not presented as validated operational localization. |
| Saved artifacts load | Verified locally | The production-configured API reported `prediction_ready: true`; required binary artifacts and optional diagnosis, attribution, and temporal artifacts loaded and were exercised. Artifacts are local/ignored and must be copied securely to the hosted volume. |
| Unified inference | Verified locally | A real normal record returned `normal` without diagnosis. A real anomalous held-out record returned `anomaly`, 11 diagnosis probabilities, attribution, and temporal localization through `/analyze`. |
| FastAPI predictions | Verified locally | Production-mode TestClient smoke exercised actual stored TelecomTS records and artifacts, required API-key behavior, readiness, model info, and analysis. An earlier local Uvicorn/API-client integration is recorded in [the integration report](./integration-test-report.md). This does not verify a hosted URL. |
| Streamlit results | Verified locally | All nine pages pass AppTest; the prior local Uvicorn integration rendered actual API analysis through the dashboard. No screenshots were captured and no public Streamlit deployment was tested. |
| Tests | Verified | 136 tests pass with the exact lock; `uv lock --check`, `uv pip check`, and Python compilation pass. See [the integration report](./integration-test-report.md). |
| Reproducible deployment | Prepared, not live-verified | `uv.lock`, runtime and test/training requirements, `.python-version`, and `render.yaml` are present. Hosted provisioning and production traffic remain operator actions. |

## Local artifact inventory

The listed model artifacts were present and loadable during the local
real-artifact smoke test. SHA-256 values identify the files checked in this
workspace; regenerate them after any model or metadata change.

| File | Bytes | SHA-256 |
|---|---:|---|
| `detection.joblib` | 4,041,081 | `490266856C6C376D9CC4B2D62A141AE2BB6FC7C5711A8D5845DF080DA2D8634E` |
| `preprocessing.joblib` | 77 | `20A3B562DD7FBF3868695E0D0E5468135043E16646A6DF287E417EEDA3D63781` |
| `feature_config.json` | 57,916 | `AA657829B03D6FFF83CB8168C85027A13D8C9DF4508872A1B983F755EE0BF2F1` |
| `model_metadata.json` | 11,930 | `C5F607C05A031D9749DBED80D8190D314F58CD5F33DF63DF65E9EE17941CA4D0` |
| `diagnosis.joblib` | 5,782,416 | `93E01AAF138ABEEA919B9F6B37D790A423CFFC2C0CEF0F80347C002F93B4069B` |
| `attribution.joblib` | 11,029 | `B9CB69E84235B25296D8649391824803A780C90C9C746877E17D1D5D40F40D26` |
| `temporal_localization.joblib` | 3,563,427 | `7C0D61F48FA2F20ADBAF06A0CBE12F8F5E36EA38B7D3C982ACC4E15287D54D15` |

Additional diagnosis class mapping, configuration, and metadata files are in
the same ignored `artifacts/` directory. Do not copy secrets into that
directory.

## Dependency and test verification

The runtime uses the exact versions in `requirements.txt`; the complete
research/test environment is exported to `requirements-test.txt`. `uv.lock`
contains the resolved dependency graph, and `.python-version` records
`3.13.3`. A clean local `uv sync --frozen --all-extras` installed the project
environment; `uv pip check` found no incompatibilities.

The final suite command was:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q
```

Result: **136 passed in 26.58 seconds**. The production-mode saved-artifact
smoke check separately verified readiness, unauthorized model-info rejection,
authorized model access, normal short-circuiting, anomaly diagnosis, class
probabilities, attribution, and temporal localization.

## Implemented versus unverified

**Implemented:** ingestion and validation, preprocessing/EDA, leakage-safe
features, group-aware splits, training/evaluation workflows, binary and
multiclass models, optional attribution and localization, unified inference,
FastAPI, nine-page Streamlit dashboard, API-key protection, deployment URL
validation, and reproducible dependency definitions.

**Experimentally validated locally:** full-dataset ingestion and EDA, grouped
split accounting, the reported model evaluations, 136 automated tests, locked
environment installation, and API inference using genuine saved artifacts
and real held-out records.

**Not verified:** deploying or operating Render/Streamlit Community Cloud,
hosted artifact provisioning, public health/prediction endpoints, account
secrets/domains, and sustained production traffic. The dashboard is not
connected to live 5G telemetry and does not provide real-time network
monitoring.

**Known research limitations:** diagnosis has poor and uneven held-out
performance; temporal annotations often cover nearly the full sequence and
the experimental localizer does not exceed the simple test baseline; the
shared final test partition is historical due to prior model evaluation.
