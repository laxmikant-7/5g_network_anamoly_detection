"""Streamlit dashboard for cross-layer 5G anomaly analysis."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cl5g.common.logging_config import configure_logging
from cl5g.data.telecomts import CATEGORICAL_KPIS, KPI_NAMES, NUMERIC_KPIS
from cl5g.pipeline.config import AppConfig, load_config
from frontend.components import (
    apply_dashboard_theme,
    empty_state,
    metric_row,
    page_header,
    plotly_chart,
)
from frontend.dashboard_data import (
    evaluation_results,
    load_saved_evaluation_reports,
    load_historical_csvs,
    numeric_kpi_columns,
    parse_kpi_sequence,
)
from frontend.status import api_base_url

_PAGES = (
    "Network Overview",
    "Dataset Exploration",
    "KPI Monitoring",
    "Anomaly Detection",
    "Anomaly Diagnosis",
    "KPI Attribution",
    "Temporal Localization",
    "Model Evaluation",
    "About the Project",
)
_PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c6d3e2", "family": "Inter, sans-serif"},
    "margin": {"l": 16, "r": 16, "t": 36, "b": 16},
}
_ACCENT = ("#35d4e5", "#388bfd", "#a78bfa", "#34d399", "#f59e0b")


def main() -> None:
    """Render the navigation shell and selected dashboard page."""
    import streamlit as st

    st.set_page_config(
        page_title="Cross-Layer 5G Anomaly Intelligence",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dashboard_theme(st)

    config = load_config()
    configure_logging(config.log_level)
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = api_base_url(config)
    if "active_api_base_url" not in st.session_state:
        st.session_state.active_api_base_url = st.session_state.api_base_url
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "latest_analysis" not in st.session_state:
        st.session_state.latest_analysis = None
    if "kpi_sequence" not in st.session_state:
        st.session_state.kpi_sequence = None
    if "evaluation_payload" not in st.session_state:
        st.session_state.evaluation_payload = None

    with st.sidebar:
        st.markdown("## 📡 Network intelligence")
        st.caption("Cross-layer 5G anomaly localization")
        selected_page = st.radio(
            "Navigate",
            _PAGES,
            label_visibility="collapsed",
        )
        st.divider()
        st.text_input("FastAPI service URL", key="api_base_url")
        st.caption("Predictions and model analysis run only through FastAPI.")
        if st.session_state.api_base_url != st.session_state.active_api_base_url:
            st.session_state.latest_prediction = None
            st.session_state.latest_analysis = None
            st.session_state.active_api_base_url = st.session_state.api_base_url
        if st.button("Refresh service status", width="stretch"):
            st.session_state.pop("service_health", None)
            st.session_state.pop("service_model_info", None)

    historical_sets, dataset_errors = load_historical_csvs(config)

    if selected_page == "Network Overview":
        _render_overview(st, config, historical_sets, dataset_errors)
    elif selected_page == "Dataset Exploration":
        _render_dataset_exploration(st, config, historical_sets, dataset_errors)
    elif selected_page == "KPI Monitoring":
        _render_kpi_monitoring(st, config)
    elif selected_page == "Anomaly Detection":
        _render_anomaly_detection(st)
    elif selected_page == "Anomaly Diagnosis":
        _render_anomaly_diagnosis(st)
    elif selected_page == "KPI Attribution":
        _render_kpi_attribution(st)
    elif selected_page == "Temporal Localization":
        _render_temporal_localization(st)
    elif selected_page == "Model Evaluation":
        _render_model_evaluation(st)
    else:
        _render_about(st)


def _render_overview(
    st: Any,
    config: AppConfig,
    historical_sets: list[tuple[str, pd.DataFrame]],
    dataset_errors: list[str],
) -> None:
    page_header(
        st,
        "Network overview",
        "A grounded view of historical data availability, saved models, and validated evaluation.",
        "Operations",
    )
    frame, source = _active_dataset(st, historical_sets)
    health = _service_health(st)
    model_info = _model_info(st)
    saved_reports, report_errors = _load_saved_reports(model_info)
    binary_report = saved_reports.get("binary_detection")

    if frame is None:
        metric_row(
            st,
            [
                ("Historical records", "Not available", None),
                ("Observed entities", "Not available", None),
                ("Observed anomaly categories", "Not available", None),
                ("Validated model performance", "Not available", None),
            ],
        )
        empty_state(
            st,
            "No historical dataset loaded",
            "Add CSV files under data/raw or data/processed, or upload a dataset on Dataset Exploration. "
            "No live network measurements are shown by this dashboard.",
            warning=True,
        )
    else:
        stats = _historical_stats(frame, config)
        metric_row(
            st,
            [
                ("Historical records", f"{stats['rows']:,}", source),
                ("Observed entities", _format_count(stats["entities"]), "From historical records only."),
                ("Observed anomaly categories", str(stats["categories"]), "Based on non-empty recorded labels."),
                (
                    "Validated model performance",
                    _evaluation_metric("accuracy", binary_report),
                    "Saved binary held-out test accuracy; unavailable if report/model versions do not match.",
                ),
            ],
        )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Historical class distribution")
        if frame is None:
            empty_state(st, "Distribution unavailable", "No historical labeled records are loaded.")
        elif config.schema.label_column not in frame.columns:
            empty_state(
                st,
                "Labels unavailable",
                f"The selected historical source does not contain '{config.schema.label_column}'.",
            )
        else:
            counts = _category_counts(frame, config)
            if counts.empty:
                empty_state(st, "No labeled records", "The selected source contains no usable anomaly labels.")
            else:
                figure = px.bar(
                    counts,
                    x="Category",
                    y="Records",
                    color="Category",
                    color_discrete_sequence=_ACCENT,
                    title="Observed labels in the selected historical source",
                )
                figure.update_layout(**_PLOT_LAYOUT, showlegend=False)
                plotly_chart(st, figure, key="overview_class_distribution")

    with right:
        st.subheader("Inference service")
        if health is None:
            empty_state(
                st,
                "Backend unavailable",
                "The FastAPI health endpoint did not respond. Model status and predictions are not assumed.",
                warning=True,
            )
        else:
            liveness = health.get("liveness_status", "unknown")
            readiness = health.get("readiness_status", "unknown")
            st.markdown(f"**Application liveness:** `{liveness}`")
            st.markdown(f"**Prediction readiness:** `{readiness}`")
            if not health.get("prediction_ready", False):
                st.warning(
                    health.get(
                        "readiness_error",
                        "Required prediction artifacts are not ready.",
                    )
                )
            components = health.get("loaded_components", {})
            if components:
                st.dataframe(
                    pd.DataFrame(
                        [{"Component": name.replace("_", " ").title(), "Loaded": loaded}
                         for name, loaded in components.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "Artifact availability and in-process loading state are service facts; "
                "they are separate from validated model performance."
            )
    if model_info is not None:
        categories = model_info.get("supported_anomaly_categories", [])
        st.caption(
            f"**Supported by the loaded diagnosis model:** "
            f"{', '.join(categories) if categories else 'not reported'}."
        )
    if binary_report is not None:
        pristine = binary_report.get("test_set_pristine")
        status = "pristine at binary evaluation" if pristine is True else "historical test partition"
        st.caption(
            f"Saved binary report: {binary_report.get('sample_count', 'unknown')} "
            f"test samples · {status}."
        )
    observed_categories = (
        sorted(
            frame[config.schema.label_column]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
            .unique()
            .tolist()
        )
        if frame is not None and config.schema.label_column in frame.columns
        else []
    )
    st.caption(
        "**Observed in selected history:** "
        + (", ".join(observed_categories) if observed_categories else "not available")
    )
    for error in report_errors:
        st.warning(f"Saved evaluation report issue: {error}")
    if dataset_errors:
        st.warning("Some historical CSV files could not be read: " + "; ".join(dataset_errors))


def _render_dataset_exploration(
    st: Any,
    config: AppConfig,
    historical_sets: list[tuple[str, pd.DataFrame]],
    dataset_errors: list[str],
) -> None:
    page_header(
        st,
        "Dataset exploration",
        "Explore historical labels and numeric KPI distributions without mixing them with live predictions.",
        "Data",
    )
    uploaded = st.file_uploader(
        "Load a historical dataset (CSV)",
        type=["csv"],
        key="historical_dataset_upload",
        help="Used for historical exploration only; this does not run inference.",
    )
    if uploaded is not None:
        try:
            frame = pd.read_csv(
                io.BytesIO(uploaded.getvalue()),
                keep_default_na=False,
            )
        except (UnicodeError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Could not read the uploaded historical dataset: {exc}")
            frame = None
        if frame is not None:
            st.session_state.historical_dataset = frame
            st.session_state.historical_dataset_source = f"Uploaded: {uploaded.name}"

    frame, source = _active_dataset(st, historical_sets)
    if frame is None:
        empty_state(
            st,
            "No historical dataset available",
            "Upload a labeled CSV above or place historical CSV files under data/raw or data/processed. "
            "The schema fixture is not treated as a measured KPI dataset.",
            warning=True,
        )
        if dataset_errors:
            st.warning("CSV read issues: " + "; ".join(dataset_errors))
        return

    st.caption(f"Historical source: **{source}** · {len(frame):,} records")
    missing_kpis = [name for name in KPI_NAMES if name not in frame.columns]
    label_column = config.schema.label_column
    with st.expander("Dataset schema validation"):
        if not missing_kpis:
            st.success("All 18 canonical TelecomTS KPI columns are present.")
        else:
            st.info(
                "This dataset does not contain the full TelecomTS KPI schema. "
                "Missing channels: " + ", ".join(missing_kpis)
            )
        if label_column in frame.columns:
            unknown_labels = sorted(
                set(frame[label_column].astype(str).str.strip())
                - set(config.schema.anomaly_types)
                - {""}
            )
            if unknown_labels:
                st.warning("Labels outside the configured taxonomy: " + ", ".join(unknown_labels))
            else:
                st.caption("Observed labels match the configured taxonomy.")
        entity_column = config.schema.entity_column
        if entity_column in frame.columns:
            per_entity_rows = frame.groupby(entity_column, dropna=False).size()
            invalid_lengths = int(per_entity_rows.ne(128).sum())
            st.caption(
                f"Sequence-group check: {int(per_entity_rows.eq(128).sum()):,} "
                f"entities have 128 rows; {invalid_lengths:,} differ."
            )
    if label_column in frame.columns:
        labels = frame[label_column].dropna().astype(str).str.strip()
        labels = labels[labels.ne("")]
        normal_count = int(labels.str.lower().eq("normal").sum())
        anomaly_count = int((~labels.str.lower().eq("normal")).sum())
        metric_row(
            st,
            [
                ("Normal records", f"{normal_count:,}", "Historical labels only."),
                ("Anomalous records", f"{anomaly_count:,}", "Historical labels only."),
                ("Distinct observed labels", str(labels.nunique()), "Derived from this dataset."),
            ],
        )
        observed = sorted(labels.unique().tolist())
        categories = list(dict.fromkeys([*config.schema.anomaly_types, *observed]))
        counts = labels.value_counts()
        distribution = pd.DataFrame(
            {"Category": categories, "Records": [int(counts.get(item, 0)) for item in categories]}
        )
        figure = px.bar(
            distribution,
            x="Category",
            y="Records",
            color="Category",
            color_discrete_sequence=_ACCENT,
            title="Normal versus anomalous distribution across configured and observed labels",
        )
        figure.update_layout(**_PLOT_LAYOUT, showlegend=False)
        plotly_chart(st, figure, key="dataset_category_distribution")
        if len(observed) < 11:
            st.info(
                f"This source contains {len(observed)} distinct labels, not 11. "
                "No unobserved anomaly categories or counts are invented."
            )
    else:
        empty_state(
            st,
            "Anomaly labels unavailable",
            f"Historical class distributions require a '{label_column}' column.",
        )

    numeric_columns = numeric_kpi_columns(frame, config)
    st.subheader("Historical KPI distributions")
    if not numeric_columns:
        empty_state(
            st,
            "No numeric KPI columns found",
            "The selected historical source contains no numeric KPI fields.",
        )
    else:
        selected = st.multiselect(
            "Numeric fields",
            numeric_columns,
            default=numeric_columns[: min(6, len(numeric_columns))],
            key="dataset_numeric_fields",
        )
        if selected:
            selected_values = frame[selected].apply(pd.to_numeric, errors="coerce")
            chart_data = selected_values.melt(var_name="KPI", value_name="Value").dropna()
            figure = px.histogram(
                chart_data,
                x="Value",
                color="KPI",
                barmode="overlay",
                opacity=0.68,
                marginal="box",
                color_discrete_sequence=_ACCENT,
                title="Observed historical KPI value distributions",
            )
            figure.update_layout(**_PLOT_LAYOUT)
            plotly_chart(st, figure, key="historical_kpi_distributions")
            box_data = selected_values.melt(var_name="KPI", value_name="Value").dropna()
            figure = px.box(
                box_data,
                x="KPI",
                y="Value",
                color="KPI",
                color_discrete_sequence=_ACCENT,
                title="Historical KPI box plots (fliers are retained)",
            )
            figure.update_layout(**_PLOT_LAYOUT, showlegend=False)
            plotly_chart(st, figure, key="historical_kpi_box_plots")
            if len(selected) > 1:
                correlation = selected_values.corr(method="pearson")
                figure = px.imshow(
                    correlation,
                    text_auto=".2f",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu",
                    title="Selected numerical KPI correlation",
                )
                figure.update_layout(**_PLOT_LAYOUT)
                plotly_chart(st, figure, key="historical_kpi_correlation")
        if label_column in frame.columns:
            st.subheader("Normal versus anomalous KPI distribution")
            comparison_kpi = st.selectbox(
                "KPI for class comparison",
                numeric_columns,
                key="dataset_class_comparison_kpi",
            )
            comparison = frame[[label_column, comparison_kpi]].dropna().copy()
            comparison = comparison[
                comparison[label_column].astype(str).str.strip().ne("")
            ]
            comparison["Class"] = np.where(
                comparison[label_column].astype(str).str.strip().str.lower().eq("normal"),
                "Normal",
                "Anomalous",
            )
            figure = px.histogram(
                comparison,
                x=comparison_kpi,
                color="Class",
                barmode="overlay",
                opacity=0.66,
                marginal="box",
                color_discrete_map={"Normal": _ACCENT[1], "Anomalous": _ACCENT[0]},
                title=f"Observed {comparison_kpi} values by historical class",
            )
            figure.update_layout(**_PLOT_LAYOUT)
            plotly_chart(st, figure, key="normal_anomaly_kpi_distribution")

    categorical_columns = [name for name in CATEGORICAL_KPIS if name in frame.columns]
    if categorical_columns:
        st.subheader("Categorical protocol distributions")
        protocol_frames = []
        for name in categorical_columns:
            counts = frame[name].astype(str).value_counts(dropna=False)
            protocol_frames.extend(
                {
                    "KPI": name,
                    "Value": value if value else "(empty)",
                    "Records": int(count),
                }
                for value, count in counts.items()
            )
        figure = px.bar(
            pd.DataFrame(protocol_frames),
            x="Value",
            y="Records",
            color="KPI",
            barmode="group",
            title="Observed protocol categories (empty strings retained)",
        )
        figure.update_layout(**_PLOT_LAYOUT)
        plotly_chart(st, figure, key="historical_protocol_distribution")

    if {"timestamp", config.schema.entity_column}.issubset(frame.columns) and numeric_columns:
        st.subheader("Historical sequence time series")
        entity_options = sorted(frame[config.schema.entity_column].astype(str).unique())
        selected_entity = st.selectbox(
            "Historical entity",
            entity_options,
            key="historical_timeseries_entity",
        )
        entity_frame = frame[
            frame[config.schema.entity_column].astype(str).eq(selected_entity)
        ].copy()
        entity_frame["_parsed_timestamp"] = pd.to_datetime(
            entity_frame["timestamp"],
            errors="coerce",
            utc=True,
        )
        entity_frame = entity_frame.dropna(subset=["_parsed_timestamp"]).sort_values(
            "_parsed_timestamp"
        )
        time_kpis = st.multiselect(
            "Time-series KPI channels",
            numeric_columns,
            default=numeric_columns[: min(3, len(numeric_columns))],
            key="historical_timeseries_kpis",
        )
        if time_kpis and not entity_frame.empty:
            time_data = entity_frame[["_parsed_timestamp", *time_kpis]].copy()
            for name in time_kpis:
                time_data[name] = pd.to_numeric(time_data[name], errors="coerce")
            time_data = time_data.melt(
                id_vars="_parsed_timestamp",
                var_name="KPI",
                value_name="Value",
            ).dropna()
            figure = px.line(
                time_data,
                x="_parsed_timestamp",
                y="Value",
                color="KPI",
                title=f"Historical KPI time series · {selected_entity}",
            )
            figure.update_layout(**_PLOT_LAYOUT, hovermode="x unified")
            plotly_chart(st, figure, key="historical_entity_timeseries")
    with st.expander("Inspect records"):
        st.dataframe(frame.head(500), width="stretch", hide_index=True)


def _render_kpi_monitoring(st: Any, config: AppConfig) -> None:
    page_header(
        st,
        "KPI monitoring",
        "Visualize the uploaded sequence. Values are user-provided and are not live network telemetry.",
        "Sequence",
    )
    values, names = _stored_sequence(st)
    if values is None:
        empty_state(
            st,
            "No KPI sequence loaded",
            "Upload a 128 × 18 CSV or JSON sequence below or on Anomaly Detection.",
        )
        values, names = _sequence_upload(st, "kpi_monitoring_uploader")
    if values is None:
        return

    names = _resolve_channel_names(st, names)
    metric_row(
        st,
        [
            ("Time steps", str(values.shape[0]), "Uploaded sequence."),
            ("KPI channels", str(values.shape[1]), "Channel names from saved feature configuration when available."),
            ("Sequence source", str(st.session_state.get("sequence_source", "Uploaded")), None),
        ],
    )
    melted = pd.DataFrame(values, columns=names)
    melted.index.name = "Time step"
    melted = melted.reset_index().melt(id_vars="Time step", var_name="KPI", value_name="Value")
    figure = px.line(
        melted,
        x="Time step",
        y="Value",
        color="KPI",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Uploaded KPI sequence across 128 time steps",
    )
    interval = (
        st.session_state.get("latest_analysis", {}).get("temporal_localization")
        if isinstance(st.session_state.get("latest_analysis"), dict)
        else None
    )
    if isinstance(interval, dict) and {"start", "end"}.issubset(interval):
        figure.add_vrect(
            x0=interval["start"],
            x1=interval["end"],
            fillcolor="#388bfd",
            opacity=0.15,
            line_width=0,
            annotation_text="Predicted interval",
        )
    figure.update_layout(**_PLOT_LAYOUT, hovermode="x unified", legend_title_text="Channel")
    plotly_chart(st, figure, key="kpi_monitoring_sequence")

    selected_kpi = st.selectbox(
        "Individual KPI channel",
        names,
        key="monitoring_individual_kpi",
    )
    selected_index = names.index(selected_kpi)
    selected_values = values[:, selected_index]
    selected_trace = pd.DataFrame(
        {
            "Time step": np.arange(values.shape[0]),
            "Value": selected_values,
        }
    )
    figure = px.line(
        selected_trace,
        x="Time step",
        y="Value",
        title=f"Uploaded sequence · {selected_kpi}",
        color_discrete_sequence=[_ACCENT[0]],
    )
    if isinstance(interval, dict) and {"start", "end"}.issubset(interval):
        figure.add_vrect(
            x0=interval["start"],
            x1=interval["end"],
            fillcolor="#388bfd",
            opacity=0.15,
            line_width=0,
            annotation_text="Predicted interval",
        )
    figure.update_layout(**_PLOT_LAYOUT, hovermode="x unified")
    plotly_chart(st, figure, key="monitoring_individual_kpi_chart")

    if selected_kpi in NUMERIC_KPIS:
        numeric_trace = pd.to_numeric(pd.Series(selected_values), errors="coerce")
        metric_row(
            st,
            [
                ("Mean", f"{numeric_trace.mean():.4g}", selected_kpi),
                ("Median", f"{numeric_trace.median():.4g}", selected_kpi),
                ("Standard deviation", f"{numeric_trace.std():.4g}", selected_kpi),
                ("Min / max", f"{numeric_trace.min():.4g} / {numeric_trace.max():.4g}", selected_kpi),
            ],
        )
    else:
        categorical_values = pd.Series(selected_values, dtype="object").fillna("(empty)")
        transitions = int(categorical_values.ne(categorical_values.shift()).iloc[1:].sum())
        st.caption(
            f"Observed values: {categorical_values.value_counts().to_dict()} · "
            f"protocol transitions: {transitions}."
        )

    numeric_names = [name for name in names if name in NUMERIC_KPIS]
    if len(numeric_names) >= 2:
        compared = st.multiselect(
            "Compare numerical KPI trends (min–max normalized)",
            numeric_names,
            default=numeric_names[: min(3, len(numeric_names))],
            key="monitoring_compare_kpis",
        )
        if compared:
            normalized = pd.DataFrame({"Time step": np.arange(values.shape[0])})
            for name in compared:
                values_for_kpi = pd.to_numeric(
                    pd.Series(values[:, names.index(name)]),
                    errors="coerce",
                )
                span = values_for_kpi.max() - values_for_kpi.min()
                normalized[name] = (
                    (values_for_kpi - values_for_kpi.min()) / span
                    if span
                    else 0.0
                )
            figure = px.line(
                normalized.melt(id_vars="Time step", var_name="KPI", value_name="Scaled value"),
                x="Time step",
                y="Scaled value",
                color="KPI",
                title="Relative KPI trends (each channel scaled independently)",
            )
            figure.update_layout(**_PLOT_LAYOUT, hovermode="x unified")
            plotly_chart(st, figure, key="monitoring_kpi_comparison")


def _render_anomaly_detection(st: Any) -> None:
    page_header(
        st,
        "Anomaly detection",
        "Submit one validated KPI sequence to the FastAPI binary prediction endpoint.",
        "Inference",
    )
    st.info(
        "Upload a CSV with exactly 128 rows and the 18 configured KPI headers "
        "in canonical order, or a 128 × 18 JSON array in that same order."
    )
    values, names = _sequence_upload(st, "anomaly_detection_upload")
    if values is None:
        return
    st.caption(
        f"Validated sequence: **{values.shape[0]} × {values.shape[1]}** · "
        f"source: **{st.session_state.get('sequence_source', 'Uploaded')}**"
    )
    if not st.button("Run binary detection", type="primary", key="run_binary_detection"):
        return
    from frontend.api_client import ApiClientError, predict_sample

    try:
        prediction = predict_sample(
            st.session_state.api_base_url,
            values.tolist(),
            kpi_names=(
                names
                if st.session_state.get("sequence_channel_names_explicit", False)
                else None
            ),
        )
    except ApiClientError as exc:
        st.error(f"Prediction unavailable: {exc}")
        return
    st.session_state.latest_prediction = prediction
    st.session_state.latest_analysis = None
    model_info = _model_info(st)
    threshold = (
        model_info.get("decision_threshold")
        if isinstance(model_info, dict)
        else None
    )
    _show_binary_result(st, prediction, threshold=threshold)


def _render_anomaly_diagnosis(st: Any) -> None:
    page_header(
        st,
        "Anomaly diagnosis",
        "Request anomaly category and confidence from the saved diagnosis model through FastAPI.",
        "Inference",
    )
    values, names = _stored_sequence(st)
    if values is None:
        empty_state(st, "Sequence required", "Upload a sequence on Anomaly Detection first.")
        return
    sampling_rate_hz = st.number_input(
        "Sampling rate (Hz)",
        min_value=0.1,
        value=10.0,
        step=0.1,
        help="TelecomTS sequences use 10 Hz. Set the actual rate for other inputs.",
        key="analysis_sampling_rate_hz",
    )
    if st.button("Run complete analysis", type="primary", key="run_complete_analysis"):
        from frontend.api_client import ApiClientError, analyze_sample

        try:
            analysis = analyze_sample(
                st.session_state.api_base_url,
                values.tolist(),
                sampling_rate_hz=float(sampling_rate_hz),
                kpi_names=(
                    names
                    if st.session_state.get("sequence_channel_names_explicit", False)
                    else None
                ),
            )
        except ApiClientError as exc:
            st.error(f"Analysis unavailable: {exc}")
            return
        st.session_state.latest_analysis = analysis
        st.session_state.latest_prediction = {
            key: analysis[key]
            for key in ("status", "anomaly_probability", "model_version")
            if key in analysis
        }
    analysis = st.session_state.get("latest_analysis")
    if not isinstance(analysis, dict):
        empty_state(st, "Analysis not run", "Run complete analysis to request a diagnosis.")
        return
    _show_analysis_summary(st, analysis)
    if analysis.get("explanation"):
        st.markdown(f"> {analysis['explanation']}")


def _render_kpi_attribution(st: Any) -> None:
    page_header(
        st,
        "KPI attribution",
        "Review KPI importance returned by the backend model explanation artifact.",
        "Explainability",
    )
    analysis = st.session_state.get("latest_analysis")
    if not isinstance(analysis, dict):
        empty_state(st, "No analysis available", "Run complete analysis from Anomaly Diagnosis first.")
        return
    _show_analysis_summary(st, analysis)
    attributions = analysis.get("attributions")
    if isinstance(attributions, dict):
        for key, title in (
            ("binary_detection", "Binary detection"),
            ("multiclass_diagnosis", "Multiclass diagnosis"),
        ):
            model_explanation = attributions.get(key)
            if not isinstance(model_explanation, dict):
                continue
            st.subheader(title)
            st.caption(
                f"TreeSHAP explanation for {model_explanation.get('predicted_class')} "
                f"(model output {model_explanation.get('prediction_probability', 0.0):.3f})."
            )
            st.dataframe(
                pd.DataFrame(model_explanation.get("top_kpis", [])[:5]),
                width="stretch",
                hide_index=True,
            )
            kpi_rows = model_explanation.get("top_kpis", [])[:5]
            if kpi_rows:
                attribution_frame = pd.DataFrame(kpi_rows)
                if {"kpi", "signed_contribution"}.issubset(attribution_frame.columns):
                    attribution_figure = px.bar(
                        attribution_frame,
                        x="signed_contribution",
                        y="kpi",
                        orientation="h",
                        color="signed_contribution",
                        color_continuous_scale=["#f59e0b", "#35d4e5"],
                        title=f"Top KPI contributions · {title}",
                    )
                    attribution_figure.update_layout(
                        **_PLOT_LAYOUT,
                        yaxis={"categoryorder": "total ascending"},
                    )
                    plotly_chart(
                        st,
                        attribution_figure,
                        key=f"attribution_{key}_kpi_chart",
                    )
            with st.expander(f"{title} feature contributions"):
                st.dataframe(
                    pd.DataFrame(model_explanation.get("top_features", [])),
                    width="stretch",
                    hide_index=True,
                )
        st.caption(
            attributions.get(
                "interpretation",
                "Attributions describe model associations, not physical causes.",
            )
        )
        if analysis.get("explanation"):
            st.markdown(f"> {analysis['explanation']}")
        return
    important = analysis.get("important_kpis")
    if not important:
        empty_state(
            st,
            "Attribution unavailable",
            "The backend did not return KPI attributions. No feature scores are estimated in the dashboard.",
        )
        return
    st.subheader("Most influential KPIs")
    st.caption(
        "The API provides an importance ordering only; magnitudes are not returned, "
        "so this view does not imply numeric attribution scores."
    )
    st.dataframe(
        pd.DataFrame(
            {"Rank": range(1, len(important) + 1), "KPI": important}
        ),
        width="stretch",
        hide_index=True,
    )
    if analysis.get("explanation"):
        st.markdown(f"> {analysis['explanation']}")


def _render_temporal_localization(st: Any) -> None:
    page_header(
        st,
        "Temporal localization",
        "Compare the uploaded KPI trace with the interval returned by the temporal model.",
        "Localization",
    )
    values, sequence_names = _stored_sequence(st)
    if values is None:
        empty_state(st, "Sequence required", "Upload a sequence on Anomaly Detection first.")
        return
    names = _resolve_channel_names(st, sequence_names)
    analysis = st.session_state.get("latest_analysis")
    if not isinstance(analysis, dict):
        empty_state(st, "Localization not available", "Run complete analysis to request a model interval.")
        return
    interval = analysis.get("temporal_localization")
    if not isinstance(interval, dict):
        empty_state(
            st,
            "Temporal interval unavailable",
            "The backend returned no temporal localization for this prediction.",
        )
    selected_index = st.selectbox(
        "KPI channel",
        range(len(names)),
        format_func=lambda index: names[index],
        key="temporal_kpi_channel",
    )
    trace = pd.DataFrame(
        {
            "Time step": np.arange(values.shape[0]),
            "KPI value": values[:, selected_index],
        }
    )
    figure = px.line(
        trace,
        x="Time step",
        y="KPI value",
        title=f"Uploaded sequence · {names[selected_index]}",
        color_discrete_sequence=[_ACCENT[0]],
    )
    if isinstance(interval, dict) and {"start", "end"}.issubset(interval):
        figure.add_vrect(
            x0=interval["start"],
            x1=interval["end"],
            fillcolor="#388bfd",
            opacity=0.2,
            line_width=0,
            annotation_text="Predicted interval",
            annotation_position="top left",
        )
    figure.update_layout(**_PLOT_LAYOUT, hovermode="x unified")
    plotly_chart(st, figure, key="temporal_localization_plot")
    if isinstance(interval, dict):
        if str(interval.get("method", "")).startswith("experimental_"):
            st.warning(
                "Experimental localization: group-held-out results did not exceed "
                "the full-window baseline. Treat this interval as research output, "
                "not a validated operational localization."
            )
        start = interval["start"]
        end = interval["end"]
        if interval.get("start_seconds") is not None:
            st.success(
                f"Model interval: steps {start}–{end} inclusive "
                f"({interval['start_seconds']:.2f}–{interval['end_seconds']:.2f} s; "
                f"{interval['duration_seconds']:.2f} s duration at "
                f"{interval['sampling_rate_hz']:g} Hz)."
            )
        else:
            st.success(f"Model interval: time steps {start}–{end} (inclusive).")
        if interval.get("localization_score") is not None:
            calibration = (
                "calibrated"
                if interval.get("score_calibrated")
                else "uncalibrated"
            )
            st.caption(
                f"Localization score: {interval['localization_score']:.3f} ({calibration})."
            )
        if interval.get("affected_kpis") is None:
            st.caption("Affected-KPI predictions are unavailable for this localizer.")
        elif interval["affected_kpis"]:
            st.write("Predicted affected KPIs: " + ", ".join(interval["affected_kpis"]))


def _render_model_evaluation(st: Any) -> None:
    page_header(
        st,
        "Model evaluation",
        "Review metrics from supplied ground-truth and prediction results; no metrics are generated from live inference.",
        "Validation",
    )
    saved_reports, report_errors = _load_saved_reports(_model_info(st))
    _render_saved_evaluation_reports(st, saved_reports, report_errors)
    st.subheader("Externally supplied labeled results")
    st.info(
        "Upload an evaluation CSV with `actual` and `predicted` columns. "
        "For a binary precision–recall curve, also include `anomaly_probability` "
        "(probability of an anomalous class)."
    )
    uploaded = st.file_uploader(
        "Evaluation results CSV",
        type=["csv"],
        key="evaluation_results_upload",
    )
    payload = st.session_state.get("evaluation_payload")
    if uploaded is not None:
        st.session_state.evaluation_payload = None
        try:
            frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            if frame.empty or not len(frame.columns):
                raise ValueError("Evaluation CSV must contain at least one labeled row and column.")
            actual_column = st.selectbox(
                "Ground-truth column",
                list(frame.columns),
                index=frame.columns.get_loc("actual") if "actual" in frame.columns else 0,
                key="evaluation_actual_column",
            )
            predicted_column = st.selectbox(
                "Predicted-label column",
                list(frame.columns),
                index=frame.columns.get_loc("predicted")
                if "predicted" in frame.columns
                else min(1, len(frame.columns) - 1),
                key="evaluation_predicted_column",
            )
            probability_options = ["None", *frame.columns.tolist()]
            default_probability = (
                probability_options.index("anomaly_probability")
                if "anomaly_probability" in probability_options
                else 0
            )
            probability_selection = st.selectbox(
                "Anomaly probability column",
                probability_options,
                index=default_probability,
                key="evaluation_probability_column",
            )
            try:
                payload = evaluation_results(
                    frame,
                    actual_column=actual_column,
                    predicted_column=predicted_column,
                    probability_column=(
                        None if probability_selection == "None" else probability_selection
                    ),
                    normal_label="normal",
                )
                payload["source"] = uploaded.name
                st.session_state.evaluation_payload = payload
            except ValueError as exc:
                payload = None
                st.error(f"Evaluation results could not be calculated: {exc}")
        except (UnicodeError, pd.errors.ParserError, ValueError) as exc:
            payload = None
            st.error(f"Could not read evaluation CSV: {exc}")

    if not isinstance(payload, dict):
        empty_state(
            st,
            "Validated evaluation results unavailable",
            "No evaluation artifact is present in this project. Upload actual labeled prediction results "
            "to render a confusion matrix, precision–recall curve, and classification report.",
            warning=True,
        )
        return
    report = payload["classification_report"]
    overall = report.get("accuracy")
    metric_row(
        st,
        [
            ("Rows evaluated", f"{payload['rows_evaluated']:,}", payload.get("source")),
            ("Accuracy", _format_percentage(overall), "Calculated from uploaded actual/predicted labels."),
            ("Weighted precision", _format_percentage(report.get("weighted avg", {}).get("precision")), None),
            ("Weighted recall", _format_percentage(report.get("weighted avg", {}).get("recall")), None),
        ],
    )
    left, right = st.columns(2, gap="large")
    with left:
        labels = payload["labels"]
        matrix = payload["confusion_matrix"]
        figure = px.imshow(
            matrix,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=["#10233a", "#176de0", "#35d4e5"],
            labels={"x": "Predicted", "y": "Actual", "color": "Records"},
            title="Confusion matrix · uploaded evaluation results",
        )
        figure.update_layout(**_PLOT_LAYOUT)
        plotly_chart(st, figure, key="evaluation_confusion_matrix")
    with right:
        if payload["precision"] is None:
            empty_state(
                st,
                "Precision–recall curve unavailable",
                "Provide an anomaly-probability column and both normal and anomalous actual labels.",
            )
        else:
            precision = payload["precision"]
            recall = payload["recall"]
            curve = go.Figure()
            curve.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    line={"color": _ACCENT[0], "width": 3},
                    name="Precision–recall",
                )
            )
            curve.update_layout(
                **_PLOT_LAYOUT,
                title="Binary precision–recall curve · uploaded results",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis={"range": [0, 1]},
                yaxis={"range": [0, 1]},
            )
            plotly_chart(st, curve, key="evaluation_precision_recall")
    st.subheader("Classification report")
    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        width="stretch",
    )
    st.caption(f"Evaluation source: {payload.get('source', 'Uploaded results')}.")


def _render_about(st: Any) -> None:
    page_header(
        st,
        "About the project",
        "Cross-Layer 5G Network Anomaly Localization and Anomaly-Type Diagnosis Using Machine Learning.",
        "Project",
    )
    st.markdown(
        """
        This dashboard provides a single interface for exploring historical KPI
        records and requesting inference from saved detection and diagnosis
        models. Uploaded sequences are sent to the configured FastAPI service;
        Streamlit does not contain a second prediction implementation and does
        not train models.

        **Data integrity**

        - Historical dataset statistics are computed only from loaded historical CSV files.
        - Uploaded sample predictions are labeled as new inference results.
        - Evaluation metrics come from saved held-out reports or separately
          identified externally supplied ground-truth and prediction records.
        - Missing model artifacts, attribution data, and temporal intervals are
          reported as unavailable rather than replaced with example values.
        - This dashboard does not connect to a live 5G network or present
          generated values as measurements.

        **Dataset and methodology**

        TelecomTS contains 128-timestep sequences with 18 canonical KPI channels
        (16 numerical and two categorical protocol channels). Source-file-aware
        partitions keep each source group together. The same saved preprocessing
        and feature-extraction contracts are used for offline training and
        online inference. Labels, timestamps, identifiers, and annotations are
        not predictive KPI inputs.

        **Research limitations**

        The dashboard's `ran_du` history label is assigned for compatibility;
        TelecomTS does not provide original layer-level ground truth, so this
        project does not claim validated cross-layer localization. KPI
        attribution describes model contribution, not causation. Temporal
        localization remains experimental and must be compared with its saved
        baseline. Historical uploads are not live telemetry.
        """
    )
    st.subheader("Dashboard workflow")
    st.markdown(
        "Historical data → KPI exploration → FastAPI binary detection → "
        "FastAPI diagnosis and localization → reviewed evaluation results"
    )
    info = _model_info(st)
    if info is not None:
        st.subheader("Configured inference features")
        st.dataframe(
            pd.DataFrame(info.get("feature_configuration", {}).get("features", [])),
            width="stretch",
            hide_index=True,
        )
    saved_reports, report_errors = _load_saved_reports(info)
    binary = saved_reports.get("binary_detection")
    diagnosis = saved_reports.get("multiclass_diagnosis")
    if binary or diagnosis:
        st.subheader("Saved evaluation evidence")
        if binary:
            metrics = binary.get("metrics", {})
            st.caption(
                f"Binary report: {binary.get('sample_count', 0):,} test samples; "
                f"pristine at binary evaluation: "
                f"{'yes' if binary.get('test_set_pristine') is True else 'not established'}."
            )
            st.write(
                f"Binary F1: {_format_percentage(metrics.get('f1'))}; "
                f"anomaly recall: {_format_percentage(metrics.get('recall'))}."
            )
        if diagnosis:
            metrics = diagnosis.get("metrics", {})
            st.caption(
                f"Diagnosis report: {diagnosis.get('sample_count', 0):,} anomaly-only "
                f"samples; historical holdout: "
                f"{'yes' if diagnosis.get('test_partition_pristine') is not True else 'no'}."
            )
            st.write(
                f"Diagnosis macro-F1: {_format_percentage(metrics.get('macro_f1'))}; "
                f"balanced accuracy: {_format_percentage(metrics.get('balanced_accuracy'))}."
            )
    for error in report_errors:
        st.warning(f"Saved evaluation report issue: {error}")


def _sequence_upload(st: Any, key: str) -> tuple[np.ndarray | None, list[str] | None]:
    uploaded = st.file_uploader(
        "Upload KPI sequence (CSV or JSON)",
        type=["csv", "json"],
        key=key,
        help="The sequence must contain exactly 128 measurements across 18 KPI channels.",
    )
    if uploaded is None:
        return None, None
    try:
        values, names = parse_kpi_sequence(uploaded.name, uploaded.getvalue())
    except ValueError as exc:
        st.session_state.kpi_sequence = None
        st.session_state.latest_prediction = None
        st.session_state.latest_analysis = None
        st.session_state.sequence_fingerprint = None
        st.error(str(exc))
        return None, None
    _store_sequence(
        st,
        values,
        names,
        uploaded.name,
        channel_names_explicit=uploaded.name.lower().endswith(".csv"),
    )
    return values, names


def _store_sequence(
    st: Any,
    values: np.ndarray,
    names: list[str],
    source: str,
    *,
    channel_names_explicit: bool,
) -> None:
    fingerprint = hashlib.sha256(
        json.dumps(
            values.tolist(),
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("sequence_fingerprint") != fingerprint:
        st.session_state.latest_prediction = None
        st.session_state.latest_analysis = None
    st.session_state.kpi_sequence = values
    st.session_state.sequence_channel_names = names
    st.session_state.sequence_channel_names_explicit = channel_names_explicit
    st.session_state.sequence_source = source
    st.session_state.sequence_fingerprint = fingerprint


def _stored_sequence(st: Any) -> tuple[np.ndarray | None, list[str] | None]:
    values = st.session_state.get("kpi_sequence")
    if values is None:
        return None, None
    return values, st.session_state.get("sequence_channel_names")


def _resolve_channel_names(st: Any, names: list[str] | None) -> list[str]:
    if names and len(names) == 18 and not all(name.startswith("Channel ") for name in names):
        return names
    info = _model_info(st)
    if info is not None:
        configured_names = info.get("feature_configuration", {}).get("kpi_names")
        if isinstance(configured_names, list) and len(configured_names) == 18:
            return [str(name) for name in configured_names]
    return names or [f"Channel {index + 1}" for index in range(18)]


def _active_dataset(
    st: Any,
    historical_sets: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame | None, str]:
    uploaded = st.session_state.get("historical_dataset")
    if isinstance(uploaded, pd.DataFrame):
        return uploaded.copy(), st.session_state.get("historical_dataset_source", "Uploaded dataset")
    if not historical_sets:
        return None, "Not available"
    if len(historical_sets) == 1:
        name, frame = historical_sets[0]
        return frame.copy(), f"data/raw or data/processed · {name}"
    combined = pd.concat(
        [frame.assign(_source=name) for name, frame in historical_sets],
        ignore_index=True,
        sort=False,
    )
    return combined, f"Combined {len(historical_sets)} CSV files from data/raw and data/processed"


def _historical_stats(frame: pd.DataFrame, config: AppConfig) -> dict[str, int | None]:
    entities: int | None = None
    if config.schema.entity_column in frame.columns:
        entities = int(frame[config.schema.entity_column].dropna().nunique())
    categories = 0
    label = config.schema.label_column
    if label in frame.columns:
        categories = int(frame[label].dropna().astype(str).str.strip().replace("", np.nan).nunique())
    return {"rows": len(frame), "entities": entities, "categories": categories}


def _category_counts(frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    label = config.schema.label_column
    if label not in frame.columns:
        return pd.DataFrame(columns=["Category", "Records"])
    labels = frame[label].dropna().astype(str).str.strip()
    labels = labels[labels.ne("")]
    categories = list(dict.fromkeys([*config.schema.anomaly_types, *sorted(labels.unique())]))
    counts = labels.value_counts()
    return pd.DataFrame(
        {"Category": categories, "Records": [int(counts.get(item, 0)) for item in categories]}
    )


def _service_health(st: Any) -> dict[str, Any] | None:
    from frontend.api_client import ApiClientError, fetch_health

    try:
        health = fetch_health(st.session_state.api_base_url)
    except ApiClientError as exc:
        st.session_state.pop("service_health", None)
        st.session_state.service_health_error = str(exc)
        return None
    st.session_state.service_health = health
    st.session_state.pop("service_health_error", None)
    return health


def _model_info(st: Any) -> dict[str, Any] | None:
    from frontend.api_client import ApiClientError, fetch_model_info

    if st.session_state.get("service_model_info_url") == st.session_state.api_base_url:
        cached = st.session_state.get("service_model_info")
        if isinstance(cached, dict):
            return cached
    try:
        info = fetch_model_info(st.session_state.api_base_url)
    except ApiClientError as exc:
        st.session_state.service_model_info_error = str(exc)
        return None
    st.session_state.service_model_info = info
    st.session_state.service_model_info_url = st.session_state.api_base_url
    st.session_state.pop("service_model_info_error", None)
    return info


def _load_saved_reports(
    model_info: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not isinstance(model_info, dict):
        return {}, []
    return load_saved_evaluation_reports(
        Path(__file__).resolve().parents[1],
        dataset_revision=model_info.get("dataset_revision"),
        model_versions=model_info.get("model_versions"),
    )


def _render_saved_evaluation_reports(
    st: Any,
    reports: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    st.subheader("Saved held-out evaluation")
    if errors:
        for error in errors:
            st.warning(f"Saved evaluation report issue: {error}")
    if not reports:
        empty_state(
            st,
            "Saved evaluation unavailable",
            "No report matching the served dataset revision and model versions is available. "
            "Live predictions are not used to calculate model performance.",
        )
        return

    binary = reports.get("binary_detection")
    if binary is not None:
        st.markdown("#### Binary anomaly detection")
        status = (
            "Test partition was pristine for this binary evaluation."
            if binary.get("test_set_pristine") is True
            else "Test partition history is not confirmed as pristine."
        )
        st.caption(
            f"Model `{binary.get('model_version', 'unknown')}` · "
            f"{binary.get('sample_count', 0):,} held-out samples · {status}"
        )
        metrics = binary.get("metrics", {})
        metric_row(
            st,
            [
                ("Accuracy", _format_percentage(metrics.get("accuracy")), None),
                ("Precision", _format_percentage(metrics.get("precision")), "Anomaly class."),
                ("Recall", _format_percentage(metrics.get("recall")), "Anomaly class."),
                ("F1", _format_percentage(metrics.get("f1")), None),
                ("F2", _format_percentage(metrics.get("f2")), "Recall-weighted F score."),
            ],
        )
        matrix = metrics.get("confusion_matrix")
        if isinstance(matrix, list) and len(matrix) == 2:
            figure = px.imshow(
                matrix,
                x=["Normal", "Anomaly"],
                y=["Normal", "Anomaly"],
                text_auto=True,
                color_continuous_scale=["#10233a", "#176de0", "#35d4e5"],
                labels={"x": "Predicted", "y": "Actual", "color": "Samples"},
                title="Binary held-out confusion matrix",
            )
            figure.update_layout(**_PLOT_LAYOUT)
            plotly_chart(st, figure, key="saved_binary_confusion_matrix")
        classification = metrics.get("classification_report")
        if isinstance(classification, dict):
            st.dataframe(
                pd.DataFrame(classification).transpose().round(4),
                width="stretch",
            )
        report_path = Path(str(binary.get("report_path", ""))).parent
        image_paths = [
            report_path / "test_precision_recall_curve.png",
            report_path / "test_roc_curve.png",
        ]
        existing_images = [path for path in image_paths if path.is_file()]
        if existing_images:
            columns = st.columns(len(existing_images))
            for column, path in zip(columns, existing_images, strict=True):
                column.image(str(path), caption=path.stem.replace("_", " ").title())
        if binary.get("test_set_pristine") is not True:
            st.warning(status)

    diagnosis = reports.get("multiclass_diagnosis")
    if diagnosis is not None:
        st.markdown("#### Multiclass anomaly diagnosis")
        pristine = diagnosis.get("test_partition_pristine")
        st.caption(
            f"Model `{diagnosis.get('model_version', 'unknown')}` · "
            f"{diagnosis.get('sample_count', 0):,} anomaly-only held-out samples."
        )
        if pristine is not True:
            history_data = diagnosis.get("test_partition_history")
            history = (
                history_data.get(
                    "description",
                    "This test partition was previously used for another evaluation.",
                )
                if isinstance(history_data, dict)
                else "This test partition was previously used for another evaluation."
            )
            st.warning(f"Historical test partition: {history}")
        metrics = diagnosis.get("metrics", {})
        metric_row(
            st,
            [
                ("Accuracy", _format_percentage(metrics.get("accuracy")), None),
                ("Balanced accuracy", _format_percentage(metrics.get("balanced_accuracy")), None),
                ("Macro F1", _format_percentage(metrics.get("macro_f1")), None),
                ("Weighted F1", _format_percentage(metrics.get("weighted_f1")), None),
            ],
        )
        confusion = metrics.get("confusion_matrix")
        if isinstance(confusion, dict):
            labels = confusion.get("labels")
            counts = confusion.get("counts")
            if isinstance(labels, list) and isinstance(counts, list):
                figure = px.imshow(
                    counts,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale=["#10233a", "#176de0", "#35d4e5"],
                    labels={"x": "Predicted", "y": "Actual", "color": "Samples"},
                    title="Multiclass held-out confusion matrix",
                )
                figure.update_layout(**_PLOT_LAYOUT, height=800)
                plotly_chart(st, figure, key="saved_multiclass_confusion_matrix")
        per_class = metrics.get("per_class")
        if isinstance(per_class, dict):
            st.dataframe(
                pd.DataFrame(per_class).transpose().round(4),
                width="stretch",
            )


def _show_binary_result(
    st: Any,
    result: dict[str, Any],
    *,
    threshold: object = None,
) -> None:
    status = result.get("status", "unknown")
    probability = result.get("anomaly_probability")
    st.subheader("New prediction")
    st.markdown(f"**Status:** `{str(status).upper()}`")
    metric_row(
        st,
        [
            ("Anomaly probability", _format_percentage(probability), "Returned by the saved detection model."),
            ("Classification threshold", _format_percentage(threshold), "Loaded from the served detector metadata."),
            ("Model version", str(result.get("model_version", "Not reported")), "From backend model metadata."),
        ],
    )
    if isinstance(probability, (int, float, np.number)) and isinstance(
        threshold,
        (int, float, np.number),
    ):
        comparison = "above" if float(probability) >= float(threshold) else "below"
        st.caption(
            f"The anomaly probability is {comparison} the saved classification threshold."
        )
    st.caption("This is a new prediction for the uploaded sequence, not a historical statistic.")


def _show_analysis_summary(st: Any, result: dict[str, Any]) -> None:
    st.subheader("Backend analysis result")
    status = result.get("status", "unknown")
    st.markdown(f"**Status:** `{str(status).upper()}`")
    metric_row(
        st,
        [
            ("Anomaly probability", _format_percentage(result.get("anomaly_probability")), None),
            ("Predicted category", str(result.get("predicted_anomaly_type") or "Not available"), None),
            ("Diagnosis confidence", _format_percentage(result.get("diagnosis_confidence")), None),
            ("Model version", str(result.get("model_version", "Not reported")), None),
        ],
    )
    probabilities = result.get("diagnosis_probabilities")
    if isinstance(probabilities, dict) and probabilities:
        st.subheader("Diagnosis class probabilities")
        distribution = pd.DataFrame(
            [
                {"Anomaly category": category, "Probability": probability}
                for category, probability in probabilities.items()
                if isinstance(category, str)
                and isinstance(probability, (int, float, np.number))
                and np.isfinite(probability)
            ]
        )
        if not distribution.empty:
            distribution = distribution.sort_values(
                "Probability",
                ascending=False,
                kind="stable",
            )
            st.dataframe(
                distribution,
                width="stretch",
                hide_index=True,
            )
            figure = px.bar(
                distribution,
                x="Anomaly category",
                y="Probability",
                color="Anomaly category",
                color_discrete_sequence=_ACCENT,
                title="Saved diagnosis model probabilities",
            )
            figure.update_layout(**_PLOT_LAYOUT, showlegend=False, yaxis={"range": [0, 1]})
            plotly_chart(st, figure, key="diagnosis_probability_distribution")


def _evaluation_metric(
    key: str,
    binary_report: dict[str, Any] | None = None,
) -> str:
    metrics = binary_report.get("metrics") if isinstance(binary_report, dict) else None
    value = metrics.get(key) if isinstance(metrics, dict) else None
    return _format_percentage(value)


def _format_percentage(value: object) -> str:
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "Not available"
    return f"{float(value):.1%}"


def _format_count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "Not available"


if __name__ == "__main__":
    main()
