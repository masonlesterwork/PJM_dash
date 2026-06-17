import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CSV_FILE = "pjm_hourly_load_zone_datetime_summed.csv"

DATETIME_COLUMN = "datetime_beginning_ept"
MKT_REGION_COLUMN = "mkt_region"
ZONE_COLUMN = "zone"
MW_COLUMN = "mw_sum"

START_DATE = "2021-01-01"
ROLLING_DAYS = 7

# ==============================================================================

def load_and_process_data():
    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()

    required = [DATETIME_COLUMN, MKT_REGION_COLUMN, ZONE_COLUMN, MW_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
        st.write("Columns found:", df.columns.tolist())
        return None

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    df[MW_COLUMN] = pd.to_numeric(df[MW_COLUMN], errors="coerce")

    df = df.dropna(subset=[DATETIME_COLUMN, MKT_REGION_COLUMN, ZONE_COLUMN, MW_COLUMN])
    df = df[df[DATETIME_COLUMN] >= pd.to_datetime(START_DATE)]

    df["year"] = df[DATETIME_COLUMN].dt.year
    df["hour"] = df[DATETIME_COLUMN].dt.hour
    df["month"] = df[DATETIME_COLUMN].dt.month

    # Remove Feb 29 to align profiles across leap/non-leap years
    df = df[~((df[DATETIME_COLUMN].dt.month == 2) & (df[DATETIME_COLUMN].dt.day == 29))]

    return df


def normalize_plot_datetime(df):
    out = df.copy()
    out["plot_datetime"] = pd.to_datetime(
        "2001-" + out[DATETIME_COLUMN].dt.strftime("%m-%d %H:%M"),
        errors="coerce",
    )
    return out


def chart_cumulative_hourly_by_year(filtered_df_for_mkt_and_zone, years):
    # filtered_df already contains exactly one zone, but we keep logic explicit
    df = filtered_df_for_mkt_and_zone.copy()
    df = normalize_plot_datetime(df)

    hourly = (
        df.groupby(["year", "plot_datetime"], as_index=False)[MW_COLUMN]
          .sum()
          .rename(columns={MW_COLUMN: "hourly_mw"})
          .sort_values(["year", "plot_datetime"])
    )

    hourly["cumulative_load"] = hourly.groupby("year")["hourly_mw"].cumsum()

    fig = go.Figure()
    for y in years:
        ydf = hourly[hourly["year"] == y]
        fig.add_trace(go.Scatter(
            x=ydf["plot_datetime"],
            y=ydf["cumulative_load"],
            mode="lines",
            name=str(y)
        ))

    zone = df[ZONE_COLUMN].iloc[0] if not df.empty else ""
    fig.update_layout(
        title=f"Cumulative Hourly Load by Year — {zone}",
        xaxis_title="Month / Day / Hour",
        yaxis_title="Cumulative Load (mw_sum-hours)",
        template="plotly_white",
        hovermode="x unified",
        height=700,
        legend=dict(title="Year"),
    )
    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig


def chart_max_mw_by_year_and_zone(filtered_df_for_years_and_mkt):
    # In this dashboard, we let the chart show across zones for the chosen market
    # but only for the filtered years.
    df = filtered_df_for_years_and_mkt.copy()

    idx = (
        df.groupby(["year", ZONE_COLUMN])[MW_COLUMN]
          .idxmax()
    )

    max_rows = df.loc[idx, ["year", ZONE_COLUMN, MW_COLUMN, DATETIME_COLUMN]].copy()
    max_rows = max_rows.rename(columns={
        MW_COLUMN: "max_mw",
        ZONE_COLUMN: "zone",
        DATETIME_COLUMN: "max_datetime",
    }).sort_values(["zone", "year"])

    zones = sorted(max_rows["zone"].unique())
    fig = go.Figure()

    for z in zones:
        sub = max_rows[max_rows["zone"] == z].sort_values("year")
        labels = sub["max_datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

        fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub["max_mw"],
            mode="lines+markers+text",
            name=str(z),
            text=labels,
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=(
                "Zone: " + str(z) +
                "<br>Year: %{x}" +
                "<br>Max MW: %{y:,.0f}" +
                "<br>Max Date/Time: %{text}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Largest Hourly Load (Max MW) by Year and Zone",
        xaxis_title="Year",
        yaxis_title="Max MW",
        template="plotly_white",
        hovermode="closest",
        height=700,
        legend_title_text="Zone",
    )
    fig.update_xaxes(dtick=1)
    return fig


def chart_max_mw_single_zone(filtered_df_for_mkt_zone, years):
    df = filtered_df_for_mkt_zone.copy()
    idx = df.groupby("year")[MW_COLUMN].idxmax()
    max_rows = df.loc[idx, ["year", MW_COLUMN, DATETIME_COLUMN]].copy()
    max_rows = max_rows.rename(columns={MW_COLUMN: "max_mw", DATETIME_COLUMN: "max_datetime"}).sort_values("year")

    labels = max_rows["max_datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    zone = df[ZONE_COLUMN].iloc[0] if not df.empty else ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=max_rows["year"],
        y=max_rows["max_mw"],
        mode="lines+markers+text",
        name=str(zone),
        text=labels,
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate=(
            f"Zone: {zone}<br>"
            "Year: %{x}<br>"
            "Max MW: %{y:,.0f}<br>"
            "Max Date/Time: %{text}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"Largest Hourly Load by Year — {zone}",
        xaxis_title="Year",
        yaxis_title="Max MW",
        template="plotly_white",
        height=700,
    )
    fig.update_xaxes(dtick=1)
    return fig


def chart_daily_heatmap(filtered_df_for_mkt_zone, years):
    df = filtered_df_for_mkt_zone.copy()
    df = df[df["year"].isin(years)]
    df["date_str"] = df[DATETIME_COLUMN].dt.strftime("%Y-%m-%d")

    agg = (
        df.groupby(["date_str", "hour"], as_index=False)[MW_COLUMN]
          .mean()
          .rename(columns={MW_COLUMN: "avg_mw"})
          .sort_values(["date_str", "hour"])
    )

    pivot = agg.pivot(index="date_str", columns="hour", values="avg_mw").sort_index()

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Viridis",
        colorbar_title="Avg MW",
    ))

    zone = df[ZONE_COLUMN].iloc[0] if not df.empty else ""
    fig.update_layout(
        title=f"Daily Load Shape Heatmap (Avg MW) — {zone}",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        template="plotly_white",
        height=650,
    )
    return fig


def chart_rolling_mean(filtered_df_for_mkt_zone, years):
    df = filtered_df_for_mkt_zone.copy()
    df = df[df["year"].isin(years)]
    df = df.sort_values(["year", DATETIME_COLUMN])

    window_hours = int(ROLLING_DAYS * 24)

    fig = go.Figure()
    zone = df[ZONE_COLUMN].iloc[0] if not df.empty else ""

    for y in years:
        ydf = df[df["year"] == y].copy()
        ydf["rolling_mean"] = ydf[MW_COLUMN].rolling(
            window=window_hours, min_periods=max(10, window_hours // 10)
        ).mean()
        ydf["rolling_std"] = ydf[MW_COLUMN].rolling(
            window=window_hours, min_periods=max(10, window_hours // 10)
        ).std()

        fig.add_trace(go.Scatter(
            x=ydf[DATETIME_COLUMN],
            y=ydf["rolling_mean"],
            mode="lines",
            name=f"{y} Rolling Mean",
        ))
        fig.add_trace(go.Scatter(
            x=ydf[DATETIME_COLUMN],
            y=ydf["rolling_std"],
            mode="lines",
            name=f"{y} Rolling Std (Vol proxy)",
            line=dict(dash="dot"),
            opacity=0.5,
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Rolling Mean / Volatility Proxy — {zone} (window {ROLLING_DAYS} days)",
        xaxis_title="Datetime",
        yaxis_title="MW / Std (proxy)",
        template="plotly_white",
        height=650,
        hovermode="x unified",
        legend=dict(title="Series"),
    )
    return fig


def chart_monthly_summary(filtered_df_for_mkt_zone, years, metric="avg"):
    df = filtered_df_for_mkt_zone.copy()
    df = df[df["year"].isin(years)]

    if metric == "max":
        agg = df.groupby(["year", "month"])[MW_COLUMN].max().reset_index(name="month_value")
        title_metric = "MAX"
    else:
        agg = df.groupby(["year", "month"])[MW_COLUMN].mean().reset_index(name="month_value")
        title_metric = "AVG"

    fig = go.Figure()
    for y in sorted(agg["year"].unique()):
        s = agg[agg["year"] == y].sort_values("month")
        fig.add_trace(go.Scatter(
            x=s["month"],
            y=s["month_value"],
            mode="lines+markers",
            name=str(y),
        ))

    zone = df[ZONE_COLUMN].iloc[0] if not df.empty else ""
    fig.update_layout(
        title=f"Month-by-Month {title_metric} Load — {zone}",
        xaxis_title="Month",
        yaxis_title=f"{metric.title()} MW",
        template="plotly_white",
        height=600,
        legend_title_text="Year",
    )
    fig.update_xaxes(dtick=1)
    return fig


# ==============================================================================
# UI
# ==============================================================================

st.set_page_config(page_title="PJM Load Dashboard", layout="wide")
st.title("PJM Hourly Load Dashboard (Aggregated)")

df = load_and_process_data()
if df is None:
    st.error(f"Could not load dataset: '{CSV_FILE}'")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

mkt_regions = sorted(df[MKT_REGION_COLUMN].dropna().unique())
default_mkt = "RTO" if "RTO" in mkt_regions else mkt_regions[0]

selected_mkt = st.sidebar.selectbox("Select Mkt Region", mkt_regions, index=mkt_regions.index(default_mkt))

zones = sorted(df[ZONE_COLUMN].dropna().unique())
default_zone = "RTO" if "RTO" in zones else zones[0]
selected_zone = st.sidebar.selectbox("Select Zone", zones, index=zones.index(default_zone))

years_available = sorted(df["year"].unique())
min_year, max_year = int(min(years_available)), int(max(years_available))

selected_years = st.sidebar.multiselect(
    "Years",
    options=list(range(min_year, max_year + 1)),
    default=list(range(min_year, max_year + 1)),
)

# Build filtered dataframes used by charts
filtered_df = df[
    (df[MKT_REGION_COLUMN] == selected_mkt) &
    (df[ZONE_COLUMN] == selected_zone) &
    (df["year"].isin(selected_years))
].copy()

filtered_df_for_mkt_years = df[
    (df[MKT_REGION_COLUMN] == selected_mkt) &
    (df["year"].isin(selected_years))
].copy()

years = sorted(filtered_df["year"].unique())

st.caption(f"Filtered rows: {len(filtered_df):,} | Mkt={selected_mkt} | Zone={selected_zone} | Years={years[:1]}..{years[-1:] if years else ''}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows", f"{len(filtered_df):,}")
with c2:
    st.metric("Years", len(years))
with c3:
    st.metric("Start", filtered_df[DATETIME_COLUMN].min().strftime("%Y-%m-%d") if not filtered_df.empty else "-")
with c4:
    st.metric("End", filtered_df[DATETIME_COLUMN].max().strftime("%Y-%m-%d") if not filtered_df.empty else "-")

tabs = st.tabs([
    "1) Cumulative Hourly (lines=years)",
    "2) Max MW (Year x Zone)",
    "3) Max MW (Selected Zone)",
    "4) Daily Heatmap",
    "5) Rolling Mean/Volatility",
    "6) Month Summary",
    "7) Raw Data"
])

with tabs[0]:
    st.subheader("Cumulative hourly load with each series = year")
    if years:
        st.plotly_chart(
            chart_cumulative_hourly_by_year(filtered_df, years),
            use_container_width=True
        )
    else:
        st.info("No data for the selected filters.")

with tabs[1]:
    st.subheader("Largest hourly load (max MW) by year and zone (for selected market)")
    if selected_years:
        st.plotly_chart(
            chart_max_mw_by_year_and_zone(filtered_df_for_mkt_years),
            use_container_width=True
        )
    else:
        st.info("Select some years.")

with tabs[2]:
    st.subheader(f"Largest hourly load by year — {selected_zone}")
    if years:
        st.plotly_chart(
            chart_max_mw_single_zone(filtered_df, years),
            use_container_width=True
        )
    else:
        st.info("No data for the selected filters.")

with tabs[3]:
    st.subheader("Daily load shape heatmap (avg MW)")
    if years:
        st.plotly_chart(
            chart_daily_heatmap(filtered_df, years),
            use_container_width=True
        )
    else:
        st.info("No data for the selected filters.")

with tabs[4]:
    st.subheader("Rolling mean / volatility proxy")
    if years:
        st.plotly_chart(
            chart_rolling_mean(filtered_df, years),
            use_container_width=True
        )
    else:
        st.info("No data for the selected filters.")

with tabs[5]:
    st.subheader("Month-by-month summary")
    metric = st.selectbox("Metric", ["avg", "max"], index=0)
    if years:
        st.plotly_chart(
            chart_monthly_summary(filtered_df, years, metric=metric),
            use_container_width=True
        )
    else:
        st.info("No data for the selected filters.")

with tabs[6]:
    st.subheader("Raw data preview (filtered)")
    st.dataframe(filtered_df.sort_values(DATETIME_COLUMN).head(5000), use_container_width=True)