"""Reusable Streamlit components and dashboard theme."""

from __future__ import annotations

from typing import Any


def apply_dashboard_theme(st: Any) -> None:
    """Apply the shared navy, blue, and cyan dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --navy-950: #07111f;
            --navy-900: #0b1728;
            --navy-800: #112238;
            --blue-500: #388bfd;
            --cyan-400: #35d4e5;
            --slate-300: #c6d3e2;
            --slate-500: #8798ad;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 82% 0%, rgba(36, 104, 170, .16), transparent 34%),
                linear-gradient(145deg, #07111f 0%, #0a1728 58%, #0c1b2d 100%);
            color: #e8f0fa;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091525 0%, #0c1b2d 100%);
            border-right: 1px solid rgba(115, 156, 196, .14);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: .35rem .25rem;
        }
        [data-testid="stMetric"] {
            background: rgba(17, 34, 56, .78);
            border: 1px solid rgba(96, 142, 185, .2);
            border-radius: 14px;
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetricLabel"] p {
            color: #9eb3c9;
            font-size: .82rem;
        }
        [data-testid="stMetricValue"] {
            color: #f0f7ff;
        }
        h1, h2, h3 { letter-spacing: -.025em; }
        .eyebrow {
            color: #35d4e5;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .page-subtitle { color: #9eb3c9; margin-top: -.5rem; }
        .panel {
            background: rgba(17, 34, 56, .68);
            border: 1px solid rgba(96, 142, 185, .18);
            border-radius: 14px;
            padding: 1rem 1.2rem;
        }
        .status-dot {
            display: inline-block;
            width: .55rem;
            height: .55rem;
            border-radius: 50%;
            margin-right: .42rem;
            background: #35d4e5;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(100deg, #176de0, #168fae);
            border: 0;
        }
        div.stButton > button[kind="primary"]:hover {
            border: 1px solid #35d4e5;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(st: Any, title: str, subtitle: str, section: str) -> None:
    """Render a consistent title block for dashboard pages."""
    st.markdown(f'<div class="eyebrow">{section}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def metric_row(st: Any, metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of metric cards with optional context."""
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics, strict=True):
        column.metric(label, value, help=help_text)


def empty_state(st: Any, title: str, description: str, *, warning: bool = False) -> None:
    """Show an explicit unavailable-data state instead of filling in defaults."""
    method = st.warning if warning else st.info
    method(f"**{title}**  \n{description}")


def plotly_chart(st: Any, figure: Any, *, key: str | None = None) -> None:
    """Apply shared responsive Plotly options."""
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
        key=key,
    )
