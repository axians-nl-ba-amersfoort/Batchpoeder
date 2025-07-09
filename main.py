import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Batchpoeder Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.image("logo.svg")

st.title("Batchpoeder Dashboard")
st.subheader("Data Overview")
st.markdown("#")
file = st.sidebar.file_uploader("Upload your dataset in excel format", type=["xlsx"])

if file is not None:
    df_raw = pd.read_excel(file, engine="openpyxl")
    if "df" not in st.session_state:
        st.session_state.df = df_raw
    st.write("Quick overview of the uploaded dataset:")
    st.dataframe(df_raw)

    st.markdown("#")
    st.subheader("Data Analysis")

    st.markdown("### Material Number Success Rate Analysis")
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
        )

        df_material = pd.merge(
            df_material,
            df_raw[["Material Number", "Material Group"]].drop_duplicates(),
            on="Material Number",
            how="left",
        )

        st.markdown("#### Success Rate by Material Number")
        st.dataframe(df_material, use_container_width=True)

        chart_material = (
            alt.Chart(df_material)
            .mark_circle(size=80)
            .encode(
                x=alt.X("Material Number:N", title="Material Number"),
                y=alt.Y("Success_rate:Q", title="Success Rate (%)"),
                color=alt.Color("Material Group:N", title="Material Group"),
                tooltip=["Material Number", "Success_rate"],
            )
            .properties(title="Success Rate by Material Number")
            .interactive()
        )

        # Use selection_point and add_params for Altair 5+
        selection = alt.selection_point(fields=["Material Number"], empty="none")
        chart_material = chart_material.add_params(selection)

        st.altair_chart(chart_material, use_container_width=True, key="material_chart")

        material_numbers = df_material["Material Number"].unique()
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
else:
    st.warning("Please upload a dataset to view the data overview.")
