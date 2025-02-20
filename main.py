import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("Analyse and Machine Learning on Red Wine Quality dataset")

df = pd.read_csv('data/winequality-red.csv', sep=';')

st.subheader("Extract of the data")
st.write(df.head())
st.subheader("Number of data (rows, columns)")
st.write(df.shape)

# Exploration of the columns
st.subheader("Exploration of the columns")
for col in df.columns:
    unique_values = df[col].unique()
    # print(f"🔹 {col} : {unique_values}")
    st.write("--------------------", col, "--------------------")
    st.write(f"🔹 number of unique value {col} : {len(unique_values)}")
    st.write(f"🔹 number of null value {col} : {df[col].isnull().sum()}")
    st.write(f"🔹 min value for {col} : {df[col].min()}")
    st.write(f"🔹 max value for {col} : {df[col].max()}")


# Data visualization

# Quality distribution
st.subheader("Data visualization")
fig, ax = plt.subplots()
ax.hist(df["quality"], bins=np.arange(df["quality"].min(), df["quality"].max() + 1, 1), color="red", alpha=0.7, edgecolor="black")
ax.set_title("Quality distribution")
ax.set_xlabel("Quality")
ax.set_ylabel("Number of wines")
st.pyplot(fig)

# Correlation matrix
st.subheader("Correlation matrix")
fig, ax = plt.subplots(figsize=(10, 10))
corr = ax.matshow(df.corr(), cmap="coolwarm")
plt.colorbar(corr)
ax.set_xticks(range(len(df.columns)))
ax.set_yticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=90)
ax.set_yticklabels(df.columns)
ax.set_title("Correlation matrix")
st.pyplot(fig)

# Characteristics boxplot of the dataset
st.subheader("Characteristics boxplot of the dataset")
selected_col = st.selectbox("Select a column", df.columns)
fig, ax = plt.subplots()
ax.boxplot(df[selected_col])
ax.set_title(f"Boxplot of {selected_col}")
st.pyplot(fig)

# Machine learning


X_test, X_train, y_test, y_train = train_test_split(df.drop('quality', axis=1), df['quality'], test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
print("Score du modèle", rf_model.score(X_test, y_test))
