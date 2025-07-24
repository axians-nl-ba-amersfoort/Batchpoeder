import streamlit as st
import pandas as pd
import altair as alt

# Main page configuration
st.set_page_config(
    page_title="Batchpoeder Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.image("logo.svg")

# Main page content
st.title("Batchpoeder Dashboard")
st.markdown("---")


file = st.sidebar.file_uploader("Upload your dataset in excel format", type=["xlsx"])


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        st.warning("Please upload a dataset to view the data overview.")
    return None


df_raw = load_data(file)
if "df" not in st.session_state:
    st.session_state.df = df_raw
st.write("Quick overview of the uploaded dataset:")
st.dataframe(df_raw)

st.markdown("---")
st.markdown("### Material Number Success Rate Analysis")
if "Material Number" in df_raw.columns and "Valuation" in df_raw.columns:
    # Calculate success rates
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
    )

    df_material = pd.merge(
        df_material,
        df_raw[["Material Number", "Material Group"]].drop_duplicates(),
        on="Material Number",
        how="left",
    )
    df_material = df_material.sort_values(by="Success_rate", ascending=False)
    st.markdown("#### Success Rate by Material Number")
    st.dataframe(df_material, use_container_width=True)

    # Add slider for filtering by success rate
    min_rate = float(df_material["Success_rate"].min())
    max_rate = float(df_material["Success_rate"].max())
    rate_range = st.slider(
        "Filter by Success Rate (%)",
        min_value=min_rate,
        max_value=max_rate,
        value=(min_rate, max_rate),
        step=0.1,
    )
    filtered_df = df_material[
        (df_material["Success_rate"] >= rate_range[0])
        & (df_material["Success_rate"] <= rate_range[1])
    ]

    # Vega-Lite chart with point selection using st.vega_lite_chart
    spec = {
        "mark": {"type": "circle", "tooltip": True},
        "params": [
            {"name": "point_selection", "select": "point"},
            {"name": "grid", "select": "interval", "bind": "scales"},
        ],
        "encoding": {
            "x": {
                "field": "Material Number",
                "type": "nominal",
                "title": "Material Number",
            },
            "y": {
                "field": "Success_rate",
                "type": "quantitative",
                "title": "Success Rate (%)",
            },
            "color": {
                "field": "Material Group",
                "type": "nominal",
                "title": "Material Group",
            },
            "tooltip": [
                {"field": "Material Number", "type": "nominal"},
                {"field": "Success_rate", "type": "quantitative"},
            ],
            "fillOpacity": {
                "condition": {"param": "point_selection", "value": 1},
                "value": 0.3,
            },
        },
        "title": "Success Rate by Material Number",
    }

    event = st.vega_lite_chart(filtered_df, spec, key="vega_chart", on_select="rerun")

    # Show the raw event result for debugging/inspection
    st.write("Selected Material Number Details:", event)

    selected_material = None
    if event and event.get("point_selection"):
        sel = event["point_selection"]
        if isinstance(sel, dict) and "Material Number" in sel:
            selected_material = sel["Material Number"]
    st.markdown("---")

    # Fallback to selectbox if nothing is selected
    if not selected_material:
        material_numbers = filtered_df["Material Number"].unique()
        selected_material = st.selectbox(
            "Select a Material Number to view details",
            options=material_numbers,
            key="material_number_select",
        )

    if selected_material:

        st.markdown(f"#### Details for Material Number: {selected_material}")
        details = (
            df_raw[df_raw["Material Number"] == selected_material]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        st.json(details)
