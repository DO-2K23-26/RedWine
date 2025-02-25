import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from redwine.data.data import df, quality_labels, acidity_labels

st.set_page_config(
    page_title="Red Wine",
    page_icon="🍷",
    # layout="wide",
)

st.title("Data visualization")

# Quality distribution
st.subheader("Quality distribution")
st.bar_chart(
    df["quality"].value_counts(),
    x_label="Quality score",
    y_label="Number of wines",
    color="#900C3F",
    horizontal=False,
)

# Correlation matrix
st.subheader("Correlation matrix")

df_corr = df.drop(["quality_cat", "fixed_acidity_cat"], axis=1)
fig, ax = plt.subplots(figsize=(10, 10))
corr = ax.matshow(df_corr.corr(), cmap="coolwarm")
plt.colorbar(corr)
ax.set_xticks(range(len(df_corr.columns)))
ax.set_yticks(range(len(df_corr.columns)))
ax.set_xticklabels(df_corr.columns, rotation=90)
ax.set_yticklabels(df_corr.columns)
ax.set_title("Correlation matrix")
st.pyplot(fig)

# Characteristics boxplot of the dataset
st.subheader("Characteristics boxplot of the dataset")
selected_col = st.selectbox("Select a column", df.columns[:-2])
fig, ax = plt.subplots()
sns.boxplot(y=df[selected_col], ax=ax, color="lightblue", showfliers=True)
ax.set_title(f"Boxplot of {selected_col}")
st.pyplot(fig)

# Boxplot by categories
st.subheader("Boxplot by categories")
col_to_plot = st.radio(
    "Select a category to compare", ["quality_cat", "fixed_acidity_cat"]
)
continuous_feature = st.selectbox(
    "Select a continuous variable for comparison",
    ["alcohol", "citric acid", "density", "pH"],
)
df_display = df.copy()
df_display["quality_cat"] = df_display["quality_cat"].replace(quality_labels)
df_display["fixed_acidity_cat"] = df_display["fixed_acidity_cat"].replace(
    acidity_labels
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    x=df_display[col_to_plot],
    y=df_display[continuous_feature],
    hue=df_display[col_to_plot],
    palette="coolwarm",
    ax=ax,
    legend=False,
)
ax.set_xlabel(col_to_plot.replace("_", " ").title())
ax.set_ylabel(continuous_feature.replace("_", " ").title())
ax.set_title(
    f"Boxplot of {continuous_feature.replace('_', ' ').title()} by {col_to_plot.replace('_', ' ').title()}"
)
st.pyplot(fig)
