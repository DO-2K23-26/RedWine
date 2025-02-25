import streamlit as st

from redwine.data.data import df

# Set page config
st.set_page_config(
    page_title="Red Wine",
    page_icon="🍷",
    layout="wide",
)

# Page Title
st.title("📋 Red Wine Data Overview")
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Data Overview")
st.sidebar.write(f"**- Rows**: {df.shape[0]}")
st.sidebar.write(f"**- Columns**: {df.shape[1]}")

# Extract of the data
st.subheader("Sample Data")
st.dataframe(df.head())

# Dataset Shape
st.subheader("Dataset Shape")
st.info(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

# Column exploration
st.subheader("Column Exploration")

# Dynamic grid layout for better spacing
num_cols = 3  # Adjust for responsiveness
cols = st.columns(num_cols)

for i, col_name in enumerate(df.columns):
    with cols[i % num_cols]:  # Distribute across columns
        st.write(f"**{col_name[0].upper() + col_name[1:]}**")
        st.write(f"- **Unique Values**: {len(df[col_name].unique())}")
        st.write(f"- **Missing Values**: {df[col_name].isnull().sum()}")
        st.write(f"- **Min Value**: {df[col_name].min()}")
        st.write(f"- **Max Value**: {df[col_name].max()}")
        st.divider()

# Explanation of categories
st.subheader("🔹 Explanation of Categories")
st.write(
    "We have categorized the quality and fixed acidity of the wines to facilitate analysis."
)

col1, col2 = st.columns(2)

# Wine Quality Categories
with col1:
    st.subheader("Wine Quality (`quality_cat`)")
    st.write("Classification of wines based on quality ratings:")
    st.write("- 🟥 **Low Quality**: `≤ 5`")
    st.write("- 🟨 **Average Quality**: `= 6`")
    st.write("- 🟩 **High Quality**: `≥ 7`")

# Fixed Acidity Categories
with col2:
    st.subheader("Fixed Acidity (`fixed_acidity_cat`)")
    st.write("Grouping wines by their fixed acidity levels:")
    st.write("- 🔵 **Low Acidity**: `< 8.0`")
    st.write("- 🟠 **Medium Acidity**: `8.0 ≤ acidity < 11.0`")
    st.write("- 🔴 **High Acidity**: `≥ 11.0`")
