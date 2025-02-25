import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from redwine.data.data import acidity_labels, df, quality_labels

st.set_page_config(
    page_title="Red Wine",
    page_icon="🍷",
    layout="wide",
)

st.title("Machine Learning - Random Forest Models")

# Sidebar
st.sidebar.header("Machine Learning Models")
st.sidebar.markdown("[Random Forest - Quality](#3487a5e8)")
st.sidebar.markdown("[XGBoost Classifier (Quality Prediction)](#459af758)")
st.sidebar.markdown("[Random Forest - Quality Category](#a3754331)")
st.sidebar.markdown("[Random Forest - Fixed Acidity](#c0649ab6)")
st.sidebar.markdown("[Clustering with KMeans](#clustering-with-kmeans)")

# Quality classification
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["quality", "quality_cat"], axis=1),
    df["quality"],
    test_size=0.2,
    random_state=42,
    stratify=df["quality"],
)
y_train = y_train.replace({3: 4})
y_test = y_test.replace({3: 4})

# print(pd.Series(y_train).value_counts())

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
)

rf_model.fit(X_train, y_train)

# Model performance
y_pred_rf = rf_model.predict(X_test)
y_test_str = y_test.replace(quality_labels)
y_pred_str = pd.Series(y_pred_rf).replace(quality_labels)


class_mapping = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}
y_train_mapped = y_train.replace(class_mapping)
y_test_mapped = y_test.replace(class_mapping)

st.subheader("🚀 XGBoost Classifier (Quality Prediction)")

col3, col4 = st.columns(2)
with col3:
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.2,
        objective="multi:softmax",
        num_class=len(class_mapping),
        random_state=42,
    )

    xgb_model.fit(X_train, y_train_mapped)

    y_pred_xgb = xgb_model.predict(X_test)
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    y_test_original = y_test_mapped.replace(reverse_mapping)
    y_pred_original = pd.Series(y_pred_xgb).replace(reverse_mapping)

    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_original, y_pred_original, output_dict=True)
        ).transpose()
    )
    st.write(
        f"**Balanced Accuracy:** {balanced_accuracy_score(y_test_original, y_pred_original):.3f}"
    )
    st.write(f"**Test Set Accuracy:** {xgb_model.score(X_test, y_test_mapped):.3f}")

    # Log Loss
    y_proba_xgb = xgb_model.predict_proba(X_test)
    try:
        st.write(f"**Log Loss:** {log_loss(y_test_mapped, y_proba_xgb):.3f}")
    except ValueError:
        st.write("⚠️ Log Loss not available: single class in y_test.")

    # Brier Score
    brier_scores_xgb = []
    for i in range(y_proba_xgb.shape[1]):
        mask = y_test_mapped == i
        if mask.sum() > 0:
            brier_scores_xgb.append(brier_score_loss(mask, y_proba_xgb[:, i]))
    st.write(f"**Brier Score:** {np.mean(brier_scores_xgb):.3f}")

with col4:
    # Confusion matrix
    st.subheader("Confusion Matrix (XGBoost)")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        confusion_matrix(y_test_original, y_pred_original),
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=sorted(y_test_original.unique()),
        yticklabels=sorted(y_test_original.unique()),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.markdown("---")
st.subheader("📊 Random Forest - Quality")
col1, col2 = st.columns(2)
with col1:
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_str, y_pred_str, output_dict=True)
        ).transpose()
    )
    st.write(f"**Balanced Accuracy:** {balanced_accuracy_score(y_test, y_pred_rf):.3f}")
    st.write(f"**Test Set Accuracy:** {rf_model.score(X_test, y_test):.3f}")

    # Log Loss
    y_proba = rf_model.predict_proba(X_test)
    try:
        st.write(f"**Log Loss:** {log_loss(y_test, y_proba):.3f}")
    except ValueError:
        st.write("⚠️ Log Loss not available: single class in y_test.")

    # Brier Score
    brier_scores = []
    for i in range(y_proba.shape[1]):
        mask = y_test == i
        if mask.sum() > 0:
            brier_scores.append(brier_score_loss(mask, y_proba[:, i]))
    st.write(f"**Brier Score:** {np.mean(brier_scores):.3f}")

with col2:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred_rf),
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=sorted(y_test.unique()),
        yticklabels=sorted(y_test.unique()),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # XGBoost Classifier
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["quality", "quality_cat"], axis=1),
        df["quality"],
        test_size=0.2,
        random_state=42,
        stratify=df["quality"],
    )


st.markdown("---")
st.subheader("📊 Random Forest - Quality Category")

cols5, cols6 = st.columns(2)
with cols5:
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["quality", "quality_cat"], axis=1),
        df["quality_cat"],
        test_size=0.2,
        random_state=42,
        stratify=df["quality_cat"],
    )
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    # Model performance
    y_pred = rf_model.predict(X_test)
    y_test_str = y_test.replace(quality_labels)
    y_pred_str = pd.Series(y_pred).replace(quality_labels)
    st.subheader("📊 Model Performance for randomForest about quality_cat")
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_str, y_pred_str, output_dict=True)
        ).transpose()
    )
    st.write(f"**Balanced Accuracy:** {balanced_accuracy_score(y_test, y_pred):.3f}")
    st.write(f"**Test Set Accuracy:** {rf_model.score(X_test, y_test):.3f}")

    # Log Loss
    y_proba = rf_model.predict_proba(X_test)
    try:
        st.write(f"**Log Loss:** {log_loss(y_test, y_proba):.3f}")
    except ValueError:
        st.write("⚠️ Log Loss not available: single class in y_test.")

    # Brier Score
    brier_scores = []
    for i in range(y_proba.shape[1]):
        mask = y_test == i
        if mask.sum() > 0:
            brier_scores.append(brier_score_loss(mask, y_proba[:, i]))
    st.write(f"**Brier Score:** {np.mean(brier_scores):.3f}")

with cols6:
    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        confusion_matrix(y_test_str, y_pred_str),
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=sorted(quality_labels.values()),
        yticklabels=sorted(quality_labels.values()),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.markdown("---")
st.subheader("📊 Random Forest - Fixed Acidity")
cols7, cols8 = st.columns(2)
with cols7:
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["fixed acidity", "fixed_acidity_cat"], axis=1),
        df["fixed_acidity_cat"],
        test_size=0.2,
        random_state=42,
        stratify=df["fixed_acidity_cat"],
    )

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_test_str = y_test.replace(acidity_labels)
    y_pred_str = pd.Series(y_pred).replace(acidity_labels)
    st.subheader("📊 Model Performance for randomForest about fixed acidity")
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_str, y_pred_str, output_dict=True)
        ).transpose()
    )
    st.write(f"**Balanced Accuracy:** {balanced_accuracy_score(y_test, y_pred):.3f}")
    st.write(f"**Test Set Accuracy:** {rf_model.score(X_test, y_test):.3f}")
    y_proba = rf_model.predict_proba(X_test)

    # Log Loss
    try:
        st.write(f"**Log Loss:** {log_loss(y_test, y_proba):.3f}")
    except ValueError:
        st.write("⚠️ Log Loss not available: single class in y_test.")

    # Brier Score
    brier_scores = []
    for i in range(y_proba.shape[1]):
        mask = y_test == i
        if mask.sum() > 0:
            brier_scores.append(brier_score_loss(mask, y_proba[:, i]))
    st.write(f"**Brier Score:** {np.mean(brier_scores):.3f}")

with cols8:
    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        confusion_matrix(y_test_str, y_pred_str),
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=sorted(acidity_labels.values()),
        yticklabels=sorted(acidity_labels.values()),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# Clustering with KMeans

st.markdown("---")
st.subheader("Clustering with KMeans")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop("quality", axis=1))

# Determine optimal number of clusters using Elbow method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(k_values, inertia, marker="o")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal K")
st.pyplot(fig)
st.write("The optimal number of clusters is hard to see, we will choose k=5")

# Apply K-Means with an optimal number of clusters (e.g., k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Scatter plot of first two principal components
st.subheader("Cluster Visualization (Alcool & Quality)")
fig, ax = plt.subplots()
sns.scatterplot(
    x=df.iloc[:, 10], y=df.iloc[:, 11], hue=df["cluster"], palette="viridis", ax=ax
)
ax.set_xlabel(df.columns[10])
ax.set_ylabel(df.columns[11])
ax.set_title("Clusters based on 11th and 12th Features")
st.pyplot(fig)

# Pairplot of clustered data
st.subheader("Pairplot of Clustered Data")
st.write("Pairplot of selected features colored by cluster")
selected_features = st.multiselect(
    "Select up to 4 features for visualization", df.columns[:-1], default=df.columns[:4]
)
if selected_features:
    pairplot_fig = sns.pairplot(
        df, vars=selected_features, hue="cluster", palette="viridis"
    )
    st.pyplot(pairplot_fig)
