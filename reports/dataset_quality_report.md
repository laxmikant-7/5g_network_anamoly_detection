# Historical TelecomTS CSV Quality Report

- Dataset revision: `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`
- CSV: `C:\Users\laxmi\Projects\cross-layer-5g-anomaly\data\processed\telecomts_historical_exploration.csv`
- SHA-256: `7521e0523417ad3e8158e3ee4b56f4e00923cfc3e5f427efb9fcf277045b7f48`
- Shape: 134,528 rows × 22 columns
- Entities: 1,051; expected 128 rows per entity
- Size: 37.415 MiB
- Overall validation: **PASS**

## Schema and integrity

- Canonical columns and ordering: pass
- Observed rows per entity: `[128]`
- Wrong-length entities: 0
- Entities with inconsistent labels: 0
- Invalid timestamps: 0
- Duplicate complete rows: 0
- Duplicate `(entity_id, timestamp)` rows: 0
- Source metadata label mapping mismatches: 0
- Anomalous samples with valid temporal annotations: 951
- Anomalous samples with valid affected-KPI annotations: 951
- Source schedule mismatches: 0

## Class distribution

| Category | Samples | Rows |
|---|---:|---:|
| Normal | 100 | 12800 |
| Jamming | 100 | 12800 |
| Co-Channel Interference (Mild) | 100 | 12800 |
| Buffer Overflow (Gradual Buildup) | 100 | 12800 |
| Co-Channel Interference (Severe) | 100 | 12800 |
| Antenna Failure | 100 | 12800 |
| Faulty RF Filters (Temporal) | 100 | 12800 |
| High Network Congestion (Gradual Buildup) | 93 | 11904 |
| Doppler Shift (Severe) | 86 | 11008 |
| Faulty Handover Algorithm (Too Frequent) | 64 | 8192 |
| Resource Allocation Bugs | 60 | 7680 |
| High Network Congestion (Sudden Spike) | 48 | 6144 |

## Numerical KPI quality and outliers

| KPI | Missing | Non-numeric | Non-finite | Unique values | Tukey outliers | Outlier rate |
|---|---:|---:|---:|---:|---:|---:|
| RSRP | 0 | 0 | 0 | 17936 | 0 | 0.0000% |
| DL_BLER | 0 | 0 | 0 | 27421 | 19749 | 14.6802% |
| DL_MCS | 0 | 0 | 0 | 55 | 23549 | 17.5049% |
| UL_BLER | 0 | 0 | 0 | 48804 | 3946 | 2.9332% |
| UL_MCS | 0 | 0 | 0 | 1051 | 9068 | 6.7406% |
| UL_NPRB | 0 | 0 | 0 | 20214 | 3849 | 2.8611% |
| UL_SNR | 0 | 0 | 0 | 19600 | 14407 | 10.7093% |
| TX_Bytes | 0 | 0 | 0 | 23305 | 0 | 0.0000% |
| RX_Bytes | 0 | 0 | 0 | 9487 | 2283 | 1.6970% |
| Estimated_UL_Buffer | 0 | 0 | 0 | 36998 | 19682 | 14.6304% |
| PRBs_DL_Current | 0 | 0 | 0 | 6700 | 0 | 0.0000% |
| PRBs_UL_Current | 0 | 0 | 0 | 3480 | 2652 | 1.9713% |
| PRB_Utilization_DL | 0 | 0 | 0 | 21906 | 0 | 0.0000% |
| PRB_Utilization_UL | 0 | 0 | 0 | 16823 | 14435 | 10.7301% |
| UL_NumberOfPackets | 0 | 0 | 0 | 8548 | 9112 | 6.7733% |
| DL_NumberOfPackets | 0 | 0 | 0 | 10979 | 8526 | 6.3377% |

Outliers are identified descriptively using the Tukey 1.5×IQR rule; they are not automatically invalid measurements.

## Categorical KPI values

- `UL_Protocol`: `{"TCP": 86891, "None": 42679, "": 3899, "UDP": 1059}`
- `DL_Protocol`: `{"TCP": 82637, "None": 46548, "": 4223, "UDP": 1120}`

## Leakage and interpretation limits

- `anomaly_type` is a target label; exclude it from model features.
- Keep all 128 rows for an `entity_id` in one group split. A row-random split leaks the same sample across partitions.
- `timestamp` and `entity_id` are identifiers/time metadata, not KPI features.
- `layer=ran_du` is assigned for dashboard compatibility and is not original TelecomTS layer ground truth.
- This capped, stratified CSV is not representative of TelecomTS class prevalence; use the complete sequence dataset for training.
- KPI associations and outlier flags are descriptive, not causal.

## Figures

- `anomaly_class_distribution.png`
- `missing_value_distribution.png`
- `kpi_correlation_heatmap.png`
- `normal_vs_anomalous_kpi_distributions.png`
- `kpi_outlier_distributions.png`

## Validation checks

- **PASS** `canonical_csv_columns`
- **PASS** `all_numeric_values_valid`
- **PASS** `categorical_values_valid`
- **PASS** `labels_in_taxonomy`
- **PASS** `entity_ids_present`
- **PASS** `exactly_128_rows_per_entity`
- **PASS** `one_label_per_entity`
- **PASS** `timestamps_valid`
- **PASS** `unique_entity_timestamps`
- **PASS** `labels_match_ingestion_metadata`
- **PASS** `timestamps_match_source_schedule`

The original CSV was read-only during this audit.
