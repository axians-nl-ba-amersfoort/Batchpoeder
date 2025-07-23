import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from statsmodels.sandbox.stats.runs import runstest_1samp
import warnings

# Suppress FutureWarning from pandas/statsmodels
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# Page and Sidebar Configuration
# ==============================================================================
st.set_page_config(
    page_title="Batch Powder Quality Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A placeholder for the logo if the file doesn't exist
try:
    st.sidebar.image("logo.svg", use_container_width=True)
except Exception:
    st.sidebar.title("Dashboard Controls")

st.title("Batch Powder Quality and Process Control Dashboard")
st.markdown(
    "An interactive tool for analyzing production data, identifying process instabilities, and performing deep-dive investigations."
)
st.markdown("---")

# ==============================================================================
# Cached Data Processing Functions
# ==============================================================================


@st.cache_data
def load_and_process_data(uploaded_file):
    """
    Loads data from an uploaded Excel file, cleans it, and engineers features.
    This function is cached to prevent reloading and reprocessing on every interaction.
    """
    df_raw = pd.read_excel(uploaded_file, engine="openpyxl")
    df = df_raw.copy()

    # --- Data Cleaning ---
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_").str.replace("[#%]", "", regex=True)
    )
    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                dup + "_" + str(i) if i != 0 else dup for i in range(sum(cols == dup))
            ]
    df.columns = cols
    if "_Rejected_Samples" in df.columns and "_Rejected_Samples_1" in df.columns:
        df.rename(
            columns={
                "_Rejected_Samples": "Rejected_Count",
                "_Rejected_Samples_1": "Rejected_Percent",
            },
            inplace=True,
        )
    df["Batch_Number"] = pd.to_numeric(
        df["Batch_Number"].astype(str).str.extract("(\d+)", expand=False),
        errors="coerce",
    )
    df.dropna(
        subset=[
            "Batch_Number",
            "Inspection_Results",
            "MIC_Description",
            "Production_Line",
            "Material_Group",
        ],
        inplace=True,
    )
    df = df.sort_values(by="Batch_Number").reset_index(drop=True)

    # --- Feature Engineering ---
    def categorize_mic(desc_series):
        conditions = [
            desc_series.str.contains("fat", case=False),
            desc_series.str.contains("protein", case=False),
            desc_series.str.contains("moisture", case=False),
            desc_series.str.contains("ash", case=False),
            desc_series.str.contains("yeasts", case=False),
            desc_series.str.contains("moulds", case=False),
            desc_series.str.contains("coag", case=False),
            desc_series.str.contains("enterobac", case=False),
            desc_series.str.contains("oxygen", case=False),
            desc_series.str.contains("total plate", case=False),
            desc_series.str.contains("taste", case=False),
            desc_series.str.contains("vacuum", case=False),
        ]
        choices = [
            "Fat",
            "Protein",
            "Moisture",
            "Ash",
            "Yeasts",
            "Moulds",
            "Coagulation",
            "Enterobacter",
            "Oxygen",
            "Total Plate Count",
            "Taste",
            "Vacuum",
        ]
        return np.select(conditions, choices, default="Other")

    def categorize_line(line_series):
        conditions = [
            line_series.str.startswith("GOR", na=False),
            line_series.str.startswith("SBR", na=False),
            line_series.str.startswith("BNV", na=False),
            line_series.str.startswith("EGRON", na=False),
            line_series.str.startswith("LINE", na=False),
        ]
        choices = ["GOR", "SBR", "BNV", "EGRON", "LINE"]
        return np.select(conditions, choices, default="Other")

    df["Main_MIC_Category"] = categorize_mic(df["MIC_Description"])
    df["Main_Production_Line"] = categorize_line(df["Production_Line"])

    return df, df_raw


def analyze_control_rules(group_df):
    """
    Analyzes a dataframe slice against 10 control chart and specification rules.
    """
    if len(group_df) < 25:
        return None
    results = {}
    series = group_df["Inspection_Results"].reset_index(drop=True)
    mean, std_dev = series.mean(), series.std()
    if std_dev == 0:
        return None

    ucl, lcl = mean + 3 * std_dev, mean - 3 * std_dev
    ucl_2s, lcl_2s = mean + 2 * std_dev, mean - 2 * std_dev
    ucl_1s, lcl_1s = mean + 1 * std_dev, mean - 1 * std_dev

    results["Rule1_Beyond_Limits"] = 1 if ((series > ucl) | (series < lcl)).any() else 0
    results["Rule2_Nine_Points_Same_Side"] = (
        1
        if ((series > mean).rolling(9).sum() >= 9).any()
        or ((series < mean).rolling(9).sum() >= 9).any()
        else 0
    )
    results["Rule3_Six_Points_Trend"] = (
        1
        if ((series.diff() > 0).rolling(5).sum() >= 5).any()
        or ((series.diff() < 0).rolling(5).sum() >= 5).any()
        else 0
    )
    results["Rule4_Fourteen_Points_Alternating"] = (
        1 if (np.sign(series.diff()).diff().abs().rolling(13).sum() >= 26).any() else 0
    )
    results["Rule5_Two_of_Three_2Sigma"] = (
        1
        if ((series > ucl_2s).rolling(3).sum() >= 2).any()
        or ((series < lcl_2s).rolling(3).sum() >= 2).any()
        else 0
    )
    results["Rule6_Four_of_Five_1Sigma"] = (
        1
        if ((series > ucl_1s).rolling(5).sum() >= 4).any()
        or ((series < lcl_1s).rolling(5).sum() >= 4).any()
        else 0
    )
    results["Rule7_Fifteen_Points_1Sigma"] = (
        1 if (series.between(lcl_1s, ucl_1s).rolling(15).sum() >= 15).any() else 0
    )
    try:
        _, p_value = runstest_1samp(series.dropna(), correction=False)
        results["Rule8_P_Test_Fail"] = 1 if p_value < 0.05 else 0
    except Exception:
        results["Rule8_P_Test_Fail"] = 0
    moving_range = series.diff().abs()
    ucl_mr = 3.267 * moving_range.mean()
    results["Rule9_Excessive_Shift"] = 1 if (moving_range > ucl_mr).any() else 0
    if (
        "Upper_Specification_Limit" in group_df.columns
        and "Lower_Specification_Limit" in group_df.columns
    ):
        out_of_spec = (
            (group_df["Inspection_Results"] > group_df["Upper_Specification_Limit"])
            | (group_df["Inspection_Results"] < group_df["Lower_Specification_Limit"])
        ).any()
        results["OutOfSpec_Violation"] = 1 if out_of_spec else 0
    else:
        results["OutOfSpec_Violation"] = 0

    results["OutOfControl_Score"] = sum(results.values())
    results["Instability_Score_%"] = round(
        (results["OutOfControl_Score"] / 10) * 100, 2
    )
    results["Total_Samples"] = len(group_df)
    return results


@st.cache_data
def calculate_pain_points(_df):
    """
    Runs the pain point analysis across all relevant data groups.
    """
    pain_point_results = []
    groups_to_check = _df.groupby(
        ["Main_MIC_Category", "Production_Line", "Material_Group"]
    )
    for (mic, line, group), group_df in groups_to_check:
        rule_results = analyze_control_rules(group_df)
        if rule_results:
            rule_results.update(
                {"MIC_Category": mic, "Production_Line": line, "Material_Group": group}
            )
            pain_point_results.append(rule_results)

    if pain_point_results:
        return (
            pd.DataFrame(pain_point_results)
            .sort_values(by="OutOfControl_Score", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame()


# ==============================================================================
# Main App Logic
# ==============================================================================
file = st.sidebar.file_uploader("Upload your dataset (XLSX format)", type=["xlsx"])

if file is None:
    st.warning("Please upload a dataset using the sidebar to begin analysis.")
    st.stop()

df, df_raw = load_and_process_data(file)
pain_points_df = calculate_pain_points(df)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Data Overview",
        "🎯 Automated Pain Point Finder",
        "📈 Interactive Pain Point Matrix",
        "🔍 Deep Dive Dashboard",
        "📋 Raw Data",
    ]
)

with tab1:
    st.header("Material Number Success Rate Analysis")
    if "Material Number" in df_raw.columns and "Valuation" in df_raw.columns:
        df_material = (
            df_raw.groupby("Material Number")
            .agg(
                total_batches=("Valuation", "count"),
                accepted_batches=("Valuation", lambda x: (x == "Accepted").sum()),
            )
            .reset_index()
        )
        df_material["Success_rate"] = (
            df_material["accepted_batches"] / df_material["total_batches"] * 100
        ).round(2)
        df_material = pd.merge(
            df_material,
            df_raw[["Material Number", "Material Group"]].drop_duplicates(),
            on="Material Number",
            how="left",
        )
        st.dataframe(df_material, use_container_width=True)
        chart_material = (
            alt.Chart(df_material)
            .mark_circle(size=100, opacity=0.7)
            .encode(
                x=alt.X("Material Number:N", title="Material Number", sort="-y"),
                y=alt.Y(
                    "Success_rate:Q",
                    title="Success Rate (%)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("Material Group:N", title="Material Group"),
                tooltip=[
                    "Material Number",
                    "Material Group",
                    "Success_rate",
                    "total_batches",
                ],
                size=alt.Size("total_batches", title="Total Batches"),
            )
            .properties(title="Success Rate by Material Number")
            .interactive()
        )
        st.altair_chart(chart_material, use_container_width=True)
    else:
        st.warning("Columns 'Material Number' or 'Valuation' not found.")

with tab2:
    st.header("Top Process Instability Pain Points")
    if not pain_points_df.empty:
        display_cols = [
            "MIC_Category",
            "Production_Line",
            "Material_Group",
            "OutOfControl_Score",
            "Instability_Score_%",
            "Total_Samples",
        ]
        st.dataframe(pain_points_df[display_cols], use_container_width=True, height=600)
    else:
        st.success("Analysis complete. No significant process instability found.")

with tab3:
    st.header("Interactive Pain Point Matrix")
    if not pain_points_df.empty:
        metric_options = {
            "Total Instability Score": "OutOfControl_Score",
            "Total Instability %": "Instability_Score_%",
            "Out of Specification": "OutOfSpec_Violation",
            "Excessive Process Shift (MR Rule)": "Rule9_Excessive_Shift",
            "Rule 1: Beyond Limits": "Rule1_Beyond_Limits",
            "Rule 2: 9 Pts on One Side": "Rule2_Nine_Points_Same_Side",
            "Rule 3: 6 Pts Trending": "Rule3_Six_Points_Trend",
            "Rule 4: 14 Pts Alternating": "Rule4_Fourteen_Points_Alternating",
            "Rule 5: 2/3 in Zone A": "Rule5_Two_of_Three_2Sigma",
            "Rule 6: 4/5 in Zone B": "Rule6_Four_of_Five_1Sigma",
            "Rule 7: 15 in Zone C": "Rule7_Fifteen_Points_1Sigma",
            "Rule 8: P-Test Failure": "Rule8_P_Test_Fail",
        }
        selected_metric_label = st.selectbox(
            "Select a metric to size the points by:",
            options=list(metric_options.keys()),
        )
        metric_col = metric_options[selected_metric_label]
        size_data = (
            pain_points_df[metric_col].astype(float) + 0.1
            if metric_col not in ["OutOfControl_Score", "Instability_Score_%"]
            else pain_points_df[metric_col]
        )
        fig_matrix = px.scatter(
            pain_points_df,
            x="Production_Line",
            y="MIC_Category",
            size=size_data,
            color="Material_Group",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hover_name="Material_Group",
            hover_data={
                "Production_Line": True,
                "MIC_Category": True,
                "Total_Samples": True,
                "OutOfControl_Score": True,
                "OutOfSpec_Violation": True,
            },
            size_max=50,
            title=f"Pain Point Matrix: Sized by {selected_metric_label}",
        )
        fig_matrix.update_layout(
            xaxis={"categoryorder": "total descending"},
            yaxis={"categoryorder": "total descending"},
            height=700,
            legend_title_text="Material Group",
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
    else:
        st.info("No pain points identified to visualize.")

with tab4:
    st.header("Cascading Deep Dive Dashboard")
    st.markdown(
        "Select a specific combination of MIC, production line, and sub-line to generate detailed I-MR control charts."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_mic = st.selectbox(
            "Main MIC:",
            ["All"] + sorted(df["Main_MIC_Category"].unique().tolist()),
            key="mic_select",
        )
    with c2:
        selected_line_cat = st.selectbox(
            "Main Line:",
            ["All"] + sorted(df["Main_Production_Line"].unique().tolist()),
            key="line_cat_select",
        )
    with c3:
        temp_df_sub_line = (
            df[df["Main_Production_Line"] == selected_line_cat]
            if selected_line_cat != "All"
            else df
        )
        selected_sub_line = st.selectbox(
            "Sub-Line:",
            ["All"] + sorted(temp_df_sub_line["Production_Line"].unique().tolist()),
            key="sub_line_select",
        )
    with c4:
        temp_df_mat = df.copy()
        if selected_mic != "All":
            temp_df_mat = temp_df_mat[temp_df_mat["Main_MIC_Category"] == selected_mic]
        if selected_sub_line != "All":
            temp_df_mat = temp_df_mat[
                temp_df_mat["Production_Line"] == selected_sub_line
            ]
        elif selected_line_cat != "All":
            temp_df_mat = temp_df_mat[
                temp_df_mat["Main_Production_Line"] == selected_line_cat
            ]
        selected_mat_group = st.selectbox(
            "Material Group:",
            ["All"] + sorted(temp_df_mat["Material_Group"].unique().tolist()),
            key="mat_group_select",
        )

    # *** NEW: Conditional display logic based on user's request ***
    if (
        selected_mic == "All"
        or selected_line_cat == "All"
        or selected_sub_line == "All"
    ):
        st.info(
            "Please select a specific value for Main MIC, Main Line, and Sub-Line to generate the deep dive charts."
        )
    else:
        # --- Filtering and Plotting Logic (executes only if specific selections are made) ---
        filtered_df = df.copy()
        if selected_mic != "All":
            filtered_df = filtered_df[filtered_df["Main_MIC_Category"] == selected_mic]
        if selected_mat_group != "All":
            filtered_df = filtered_df[
                filtered_df["Material_Group"] == selected_mat_group
            ]
        if selected_sub_line != "All":
            filtered_df = filtered_df[
                filtered_df["Production_Line"] == selected_sub_line
            ]
        elif selected_line_cat != "All":
            filtered_df = filtered_df[
                filtered_df["Main_Production_Line"] == selected_line_cat
            ]

        if len(filtered_df) < 25:
            st.warning(
                "Not enough data for a meaningful control chart (requires at least 25 data points for the selected filter)."
            )
        else:
            filtered_df = filtered_df.reset_index(drop=True)
            filtered_df["Moving_Range"] = filtered_df["Inspection_Results"].diff().abs()
            q_low, q_high = filtered_df["Inspection_Results"].quantile(
                0.01
            ), filtered_df["Inspection_Results"].quantile(0.99)
            calc_df = filtered_df[
                (filtered_df["Inspection_Results"] >= q_low)
                & (filtered_df["Inspection_Results"] <= q_high)
            ]
            if len(calc_df) < 5:
                calc_df = filtered_df.copy()

            zone_mean, zone_std = (
                calc_df["Inspection_Results"].mean(),
                calc_df["Inspection_Results"].std(),
            )
            ucl, lcl = zone_mean + 3 * zone_std, max(0, zone_mean - 3 * zone_std)
            ucl_2s, lcl_2s = zone_mean + 2 * zone_std, zone_mean - 2 * zone_std
            ucl_1s, lcl_1s = zone_mean + 1 * zone_std, zone_mean - 1 * zone_std
            mr_bar = calc_df["Moving_Range"].mean()
            ucl_mr = 3.267 * mr_bar

            series = filtered_df["Inspection_Results"]
            variation_col = pd.Series(
                ["Noise (Common Cause)"] * len(series), index=series.index
            )
            for i in (
                (series.between(lcl_1s, ucl_1s))
                .rolling(15, min_periods=15)
                .sum()[lambda x: x >= 15]
                .index
            ):
                variation_col.loc[i - 14 : i] = "Rule 7: 15 in Zone C"
            for i in (
                np.sign(series.diff()) != np.sign(series.diff().shift(1))
            ).rolling(13, min_periods=13).sum()[lambda x: x >= 13].index + 1:
                variation_col.loc[i - 13 : i] = "Rule 4: 14 Pts Alternating"
            for i in (
                series.diff()
                .gt(0)
                .rolling(5, min_periods=5)
                .sum()[lambda x: x >= 5]
                .index
                + 1
            ):
                variation_col.loc[i - 5 : i] = "Rule 3: 6 Pts Trending"
            for i in (
                series.diff()
                .lt(0)
                .rolling(5, min_periods=5)
                .sum()[lambda x: x >= 5]
                .index
                + 1
            ):
                variation_col.loc[i - 5 : i] = "Rule 3: 6 Pts Trending"
            for i in (
                (series > zone_mean)
                .rolling(9, min_periods=9)
                .sum()[lambda x: x >= 9]
                .index
            ):
                variation_col.loc[i - 8 : i] = "Rule 2: 9 Pts on One Side"
            for i in (
                (series < zone_mean)
                .rolling(9, min_periods=9)
                .sum()[lambda x: x >= 9]
                .index
            ):
                variation_col.loc[i - 8 : i] = "Rule 2: 9 Pts on One Side"
            for i in (series > ucl_1s).rolling(5).sum()[lambda x: x >= 4].index:
                variation_col.loc[i - 4 : i] = "Rule 6: 4/5 in Zone B"
            for i in (series < lcl_1s).rolling(5).sum()[lambda x: x >= 4].index:
                variation_col.loc[i - 4 : i] = "Rule 6: 4/5 in Zone B"
            for i in (series > ucl_2s).rolling(3).sum()[lambda x: x >= 2].index:
                variation_col.loc[i - 2 : i] = "Rule 5: 2/3 in Zone A"
            for i in (series < lcl_2s).rolling(3).sum()[lambda x: x >= 2].index:
                variation_col.loc[i - 2 : i] = "Rule 5: 2/3 in Zone A"
            variation_col.loc[(series > ucl) | (series < lcl)] = "Rule 1: Beyond Limits"
            variation_col.loc[filtered_df["Moving_Range"] > ucl_mr] = (
                "Excessive Process Shift (MR Rule)"
            )
            if "Upper_Specification_Limit" in filtered_df.columns:
                variation_col.loc[
                    (
                        filtered_df["Inspection_Results"]
                        > filtered_df["Upper_Specification_Limit"]
                    )
                    | (
                        filtered_df["Inspection_Results"]
                        < filtered_df["Lower_Specification_Limit"]
                    )
                ] = "Out of Specification"

            plot_df = filtered_df.copy()
            plot_df.loc[:, "Plot_Sequence"] = range(len(plot_df))
            plot_df.loc[:, "Variation_Type"] = variation_col

            color_map = {
                "Out of Specification": "black",
                "Excessive Process Shift (MR Rule)": "#A020F0",
                "Rule 1: Beyond Limits": "#E51A1A",
                "Rule 2: 9 Pts on One Side": "#6A3D9A",
                "Rule 3: 6 Pts Trending": "#32CD32",
                "Rule 4: 14 Pts Alternating": "#1F78B4",
                "Rule 5: 2/3 in Zone A": "#FF7F00",
                "Rule 6: 4/5 in Zone B": "#FFD700",
                "Rule 7: 15 in Zone C": "#B0C4DE",
                "Noise (Common Cause)": "royalblue",
            }

            i_fig = go.Figure()
            i_fig.add_hrect(
                y0=lcl_1s,
                y1=ucl_1s,
                fillcolor="green",
                opacity=0.1,
                layer="below",
                annotation_text="Zone C",
                annotation_position="right",
            )
            i_fig.add_hrect(
                y0=ucl_1s,
                y1=ucl_2s,
                fillcolor="yellow",
                opacity=0.15,
                layer="below",
                annotation_text="Zone B",
                annotation_position="right",
            )
            i_fig.add_hrect(
                y0=lcl_2s, y1=lcl_1s, fillcolor="yellow", opacity=0.15, layer="below"
            )
            i_fig.add_hrect(
                y0=ucl_2s,
                y1=ucl,
                fillcolor="red",
                opacity=0.15,
                layer="below",
                annotation_text="Zone A",
                annotation_position="right",
            )
            i_fig.add_hrect(
                y0=lcl, y1=lcl_2s, fillcolor="red", opacity=0.15, layer="below"
            )
            i_fig.add_trace(
                go.Scatter(
                    x=plot_df["Plot_Sequence"],
                    y=plot_df["Inspection_Results"],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    showlegend=False,
                    hoverinfo="none",
                )
            )
            for rule, color in color_map.items():
                if not plot_df[plot_df["Variation_Type"] == rule].empty:
                    i_fig.add_trace(
                        go.Scatter(
                            x=plot_df[plot_df["Variation_Type"] == rule][
                                "Plot_Sequence"
                            ],
                            y=plot_df[plot_df["Variation_Type"] == rule][
                                "Inspection_Results"
                            ],
                            mode="markers",
                            marker=dict(color=color, size=7),
                            name=rule,
                            hovertext=[
                                f"Batch: {r.Batch_Number}<br>Result: {r.Inspection_Results:.2f}<br>Type: {r.Variation_Type}"
                                for _, r in plot_df[
                                    plot_df["Variation_Type"] == rule
                                ].iterrows()
                            ],
                            hoverinfo="text",
                        )
                    )
            i_fig.add_hline(
                y=ucl, line_dash="dash", line_color="firebrick", annotation_text="UCL"
            )
            i_fig.add_hline(
                y=lcl, line_dash="dash", line_color="firebrick", annotation_text="LCL"
            )
            i_fig.add_hline(
                y=zone_mean,
                line_dash="dash",
                line_color="green",
                annotation_text="Mean",
            )
            if "Upper_Specification_Limit" in plot_df.columns:
                i_fig.add_trace(
                    go.Scatter(
                        x=plot_df["Plot_Sequence"],
                        y=plot_df["Upper_Specification_Limit"],
                        mode="lines",
                        line=dict(color="black", dash="solid", width=2),
                        name="USL (Spec Limit)",
                    )
                )
            if "Lower_Specification_Limit" in plot_df.columns:
                i_fig.add_trace(
                    go.Scatter(
                        x=plot_df["Plot_Sequence"],
                        y=plot_df["Lower_Specification_Limit"],
                        mode="lines",
                        line=dict(color="black", dash="solid", width=2),
                        name="LSL (Spec Limit)",
                    )
                )
            i_fig.update_layout(
                height=600,
                title=f"Individuals (I) Chart: {selected_mic} | {selected_line_cat} -> {selected_sub_line} | {selected_mat_group}",
                xaxis_title="Plot Sequence",
                yaxis_title="Inspection Result",
                legend_title="Variation Type",
            )
            st.plotly_chart(i_fig, use_container_width=True)

            mr_fig = px.line(
                plot_df.dropna(subset=["Moving_Range"]),
                x="Plot_Sequence",
                y="Moving_Range",
                title="Moving Range (MR) Chart",
                markers=True,
            )
            mr_fig.update_traces(marker=dict(size=5), line=dict(width=1))
            mr_fig.add_hline(
                y=ucl_mr, line_dash="dash", line_color="red", annotation_text="UCL (MR)"
            )
            mr_fig.add_hline(
                y=mr_bar,
                line_dash="dash",
                line_color="green",
                annotation_text="Average MR",
            )
            st.plotly_chart(mr_fig, use_container_width=True)

with tab5:
    st.header("Raw Data Viewer")
    st.dataframe(df_raw)
