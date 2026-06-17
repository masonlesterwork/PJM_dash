import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CSV_FILE = "pjm_hourly_load_data.csv"

DATETIME_COLUMN = "datetime_beginning_ept"
LOAD_AREA_COLUMN = "load_area"
MW_COLUMN = "mw"

START_DATE = "2021-01-01"

ROLLING_DAYS = 7  # rolling window in days -> hours


# ==============================================================================
# Data loading / processing
# ==============================================================================

def load_and_process_data():
    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()

    required = [DATETIME_COLUMN, LOAD_AREA_COLUMN, MW_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
        st.write("Columns found:", df.columns.tolist())
        return None

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    df[MW_COLUMN] = pd.to_numeric(df[MW_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATETIME_COLUMN, LOAD_AREA_COLUMN, MW_COLUMN])

    df = df[df[DATETIME_COLUMN] >= pd.to_datetime(START_DATE)]

    df["year"] = df[DATETIME_COLUMN].dt.year
    df["hour"] = df[DATETIME_COLUMN].dt.hour
    df["month"] = df[DATETIME_COLUMN].dt.month

    # Remove Feb 29 so profiles line up cleanly across leap/non-leap years
    df = df[~((df[DATETIME_COLUMN].dt.month == 2) & (df[DATETIME_COLUMN].dt.day == 29))]

    return df


def normalize_plot_datetime(df):
    # overlay year profiles on same axis by using a fake constant year
    out = df.copy()
    out["plot_datetime"] = pd.to_datetime(
        "2001-" + out[DATETIME_COLUMN].dt.strftime("%m-%d %H:%M"),
        errors="coerce"
    )
    return out


# ==============================================================================
# Charts
# ==============================================================================

def chart_cumulative_hourly_by_year(df, selected_area, years):
    sub = df[df[LOAD_AREA_COLUMN] == selected_area].copy()
    sub = normalize_plot_datetime(sub)

    hourly = (
        sub.groupby(["year", "plot_datetime"], as_index=False)[MW_COLUMN]
        .sum()
        .rename(columns={MW_COLUMN: "hourly_mw"})
        .sort_values(["year", "plot_datetime"])
    )

    hourly["cumulative_load"] = hourly.groupby("year")["hourly_mw"].cumsum()

    fig = go.Figure()
    for y in years:
        ydf = hourly[hourly["year"] == y]
        fig.add_trace(
            go.Scatter(
                x=ydf["plot_datetime"],
                y=ydf["cumulative_load"],
                mode="lines",
                name=str(y),
            )
        )

    fig.update_layout(
        title=f"Cumulative Hourly Load by Year — {selected_area}",
        xaxis_title="Month / Day / Hour",
        yaxis_title="Cumulative Load (MW-hours)",
        template="plotly_white",
        hovermode="x unified",
        height=700,
        legend=dict(title="Year"),
    )
    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig


def chart_max_mw_by_year_and_area(df, years):
    idx = (
        df[df["year"].isin(years)]
        .groupby(["year", LOAD_AREA_COLUMN])[MW_COLUMN]
        .idxmax()
    )

    max_rows = df.loc[idx, ["year", LOAD_AREA_COLUMN, MW_COLUMN, DATETIME_COLUMN]].copy()
    max_rows = max_rows.rename(
        columns={
            MW_COLUMN: "max_mw",
            LOAD_AREA_COLUMN: "load_area",
            DATETIME_COLUMN: "max_datetime",
        }
    ).sort_values(["load_area", "year"])

    areas = sorted(max_rows["load_area"].unique())
    fig = go.Figure()

    for area in areas:
        sub = max_rows[max_rows["load_area"] == area].sort_values("year")
        labels = sub["max_datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

        fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["max_mw"],
                mode="lines+markers+text",
                name=str(area),
                text=labels,
                textposition="top center",
                textfont=dict(size=10),
                hovertemplate=(
                    "Load Area: " + str(area) +
                    "<br>Year: %{x}" +
                    "<br>Max MW: %{y:,.0f}" +
                    "<br>Max Date/Time: %{text}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Largest Hourly Load (Max MW) by Year and Load Area",
        xaxis_title="Year",
        yaxis_title="Max MW",
        template="plotly_white",
        hovermode="closest",
        height=700,
        legend_title_text="Load Area",
    )
    fig.update_xaxes(dtick=1)
    return fig


def chart_max_mw_single_area(df, selected_area, years):
    sub = df[(df[LOAD_AREA_COLUMN] == selected_area) & (df["year"].isin(years))].copy()
    idx = sub.groupby("year")[MW_COLUMN].idxmax()
    max_rows = sub.loc[idx, ["year", MW_COLUMN, DATETIME_COLUMN]].copy()
    max_rows = max_rows.rename(
        columns={MW_COLUMN: "max_mw", DATETIME_COLUMN: "max_datetime"}
    ).sort_values("year")

    labels = max_rows["max_datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=max_rows["year"],
            y=max_rows["max_mw"],
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=10),
            name=str(selected_area),
            hovertemplate=(
                f"Load Area: {selected_area}<br>"
                "Year: %{x}<br>"
                "Max MW: %{y:,.0f}<br>"
                "Max Date/Time: %{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Largest Hourly Load by Year — {selected_area}",
        xaxis_title="Year",
        yaxis_title="Max MW",
        template="plotly_white",
        height=700,
    )
    fig.update_xaxes(dtick=1)
    return fig


def chart_daily_load_shape_heatmap(df, selected_area, years):
    sub = df[(df[LOAD_AREA_COLUMN] == selected_area) & (df["year"].isin(years))].copy()

    sub["date_str"] = sub[DATETIME_COLUMN].dt.strftime("%Y-%m-%d")

    agg = (
        sub.groupby(["date_str", "hour"], as_index=False)[MW_COLUMN]
        .mean()
        .rename(columns={MW_COLUMN: "avg_mw"})
        .sort_values(["date_str", "hour"])
    )

    pivot = agg.pivot(index="date_str", columns="hour", values="avg_mw").sort_index()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar_title="Avg MW",
        )
    )
    fig.update_layout(
        title=f"Daily Load Shape Heatmap (Avg MW) — {selected_area}",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        template="plotly_white",
        height=650,
    )
    return fig


def chart_rolling_mean_volatility(df, selected_area, years):
    sub = df[(df[LOAD_AREA_COLUMN] == selected_area) & (df["year"].isin(years))].copy()
    sub = sub.sort_values(["year", DATETIME_COLUMN])

    window_hours = int(ROLLING_DAYS * 24)

    fig = go.Figure()
    for y in years:
        ydf = sub[sub["year"] == y].copy()

        ydf["rolling_mean"] = ydf[MW_COLUMN].rolling(
            window=window_hours, min_periods=max(10, window_hours // 10)
        ).mean()
        ydf["rolling_std"] = ydf[MW_COLUMN].rolling(
            window=window_hours, min_periods=max(10, window_hours // 10)
        ).std()

        fig.add_trace(
            go.Scatter(
                x=ydf[DATETIME_COLUMN],
                y=ydf["rolling_mean"],
                mode="lines",
                name=f"{y} Rolling Mean",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ydf[DATETIME_COLUMN],
                y=ydf["rolling_std"],
                mode="lines",
                name=f"{y} Rolling Std (Vol proxy)",
                line=dict(dash="dot"),
                opacity=0.5,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Rolling Mean / Volatility Proxy — {selected_area} (window {ROLLING_DAYS} days)",
        xaxis_title="Datetime",
        yaxis_title="MW / Std (proxy)",
        template="plotly_white",
        height=650,
        hovermode="x unified",
        legend=dict(title="Series"),
    )
    return fig


def chart_monthly_summary(df, selected_area, years, metric="avg"):
    sub = df[(df[LOAD_AREA_COLUMN] == selected_area) & (df["year"].isin(years))].copy()

    if metric == "max":
        agg = sub.groupby(["year", "month"])[MW_COLUMN].max().reset_index(name="month_value")
    else:
        agg = sub.groupby(["year", "month"])[MW_COLUMN].mean().reset_index(name="month_value")

    fig = go.Figure()
    for y in sorted(agg["year"].unique()):
        s = agg[agg["year"] == y].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=s["month"],
                y=s["month_value"],
                mode="lines+markers",
                name=str(y),
            )
        )

    fig.update_layout(
        title=f"Month-by-Month {metric.upper()} Load — {selected_area}",
        xaxis_title="Month",
        yaxis_title=f"{metric.title()} MW",
        template="plotly_white",
        height=600,
        legend_title_text="Year",
    )
    fig.update_xaxes(dtick=1)
    return fig


# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(page_title="PJM Load Dashboard", layout="wide")
st.title("PJM Hourly Load Dashboard")

df = load_and_process_data()
if df is None:
    st.error("Could not load dataset. Check that your CSV exists and columns match.")
    st.stop()

load_areas = sorted(df[LOAD_AREA_COLUMN].dropna().unique())

default_area = "RTO" if "RTO" in load_areas else load_areas[0]
selected_area = st.sidebar.selectbox("Select Load Area", load_areas, index=load_areas.index(default_area))
years_available = sorted(df["year"].unique())
min_year, max_year = int(min(years_available)), int(max(years_available))

selected_years = st.sidebar.multiselect(
    "Years",
    options=list(range(min_year, max_year + 1)),
    default=list(range(min_year, max_year + 1)),
)

filtered_df = df[(df["year"].isin(selected_years)) & (df[LOAD_AREA_COLUMN] == selected_area)].copy()
years = sorted(filtered_df["year"].unique())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows (filtered)", f"{len(filtered_df):,}")
with c2:
    st.metric("Years", len(years))
with c3:
    st.metric("Start Date", filtered_df[DATETIME_COLUMN].min().strftime("%Y-%m-%d") if not filtered_df.empty else "-")
with c4:
    st.metric("End Date", filtered_df[DATETIME_COLUMN].max().strftime("%Y-%m-%d") if not filtered_df.empty else "-")

tabs = st.tabs([
    "1) Cumulative Hourly (by year)",
    "2) Max MW (Year x Area)",
    "3) Max MW (Selected Area)",
    "4) Daily Heatmap (avg MW)",
    "5) Rolling Mean / Volatility",
    "6) Month Summary",
    "7) Raw Data"
])

# Tab 1
with tabs[0]:
    st.subheader("Cumulative hourly load with each series = year")
    st.plotly_chart(
        chart_cumulative_hourly_by_year(df, selected_area, years),
        use_container_width=True
    )

# Tab 2
with tabs[1]:
    st.subheader("Largest hourly MW by year and load area (labels = max timestamp)")
    df_years = df[df["year"].isin(years)]
    st.plotly_chart(
        chart_max_mw_by_year_and_area(df_years, years),
        use_container_width=True
    )

# Tab 3
with tabs[2]:
    st.subheader("Largest hourly MW by year — selected load area")
    st.plotly_chart(
        chart_max_mw_single_area(df, selected_area, years),
        use_container_width=True
    )

# Tab 4
with tabs[3]:
    st.subheader("Daily load shape heatmap (avg MW by date vs hour-of-day)")
    st.plotly_chart(
        chart_daily_load_shape_heatmap(df, selected_area, years),
        use_container_width=True
    )

# Tab 5
with tabs[4]:
    st.subheader(f"Rolling mean and volatility proxy — window {ROLLING_DAYS} days")
    st.plotly_chart(
        chart_rolling_mean_volatility(df, selected_area, years),
        use_container_width=True
    )

# Tab 6
with tabs[5]:
    st.subheader("Month-by-month summary")
    metric = st.selectbox("Metric", ["avg", "max"], index=0)
    st.plotly_chart(
        chart_monthly_summary(df, selected_area, years, metric=metric),
        use_container_width=True
    )

# Tab 7
with tabs[6]:
    st.subheader("Raw data preview")
    preview = filtered_df.sort_values(DATETIME_COLUMN).head(5000)
    st.dataframe(preview, use_container_width=True)

    st.download_button(
        label="Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_pjm_hourly_load_data.csv",
        mime="text/csv",
    )