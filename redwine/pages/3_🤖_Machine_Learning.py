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
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from redwine.data.data import acidity_labels, df, quality_labels

st.set_page_config(
    page_title="Red Wine",
    page_icon="🍷",
    layout="wide",
)

st.title("Machine Learning")

# Sidebar
# st.sidebar.header("Machine Learning Models")
# st.sidebar.markdown("[Random Forest - Quality](#3487a5e8)")
# st.sidebar.markdown("[XGBoost Classifier (Quality Prediction)](#459af758)")
# st.sidebar.markdown("[Random Forest - Quality Category](#a3754331)")
# st.sidebar.markdown("[Random Forest - Fixed Acidity](#c0649ab6)")
# st.sidebar.markdown("[Clustering with KMeans](#clustering-with-kmeans)")


# Quality classification
st.subheader("Logistic Regression - Quality Category")
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["quality", "quality_cat"], axis=1),
    df["quality_cat"],
    test_size=0.2,
    random_state=42,
    stratify=df["quality"],
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(
    multi_class="multinomial", max_iter=1000, random_state=42, class_weight="balanced"
)

log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

st.subheader("Classification Report & Confusion Matrix")
col1, col2 = st.columns(2)
with col1:
    st.write(
        pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                target_names=["Bad", "Average", "Good"],
                output_dict=True,
            )
        ).transpose()
    )
with col2:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Bad", "Average", "Good"],
        yticklabels=["Bad", "Average", "Good"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.subheader("Feature Importance (Coefficients) for Class='Good'")

class_index = 2  # index of good in the dictionnary
coeffs_good = pd.Series(log_reg.coef_[class_index], index=X_train.columns).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
coeffs_good.plot(kind="bar", color="blue", alpha=0.7, ax=ax)
ax.set_title("Logistic Regression Coefficients (Class='Good')")
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient Value")
ax.axhline(0, color="black", linewidth=1)
st.pyplot(fig)

st.write(
    """
**Interpretation**:
- Positive coefficients → Increase probability of 'Good' wines.
- Negative coefficients → Decrease probability of 'Good' wines.
- Close to zero → Little effect on classification.
"""
)

st.subheader("Boxplot of Predicted Probabilities for 'Good' Wines")

y_proba = log_reg.predict_proba(X_test_scaled)
class_names = log_reg.classes_
good_index = np.where(class_names == 1)[0][0]

df_proba = pd.DataFrame(
    {"Prob_Good": y_proba[:, good_index], "True_Class": y_test.replace(quality_labels)}
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="True_Class", y="Prob_Good", data=df_proba, palette="coolwarm", ax=ax)
ax.set_title("Predicted Probability for 'Good' by True Class")
ax.set_xlabel("True Class")
ax.set_ylabel("Probability of 'Good'")
st.pyplot(fig)

st.write(
    """
**Interpretation**:
- Ideally, *Bad* → prob(Good) basse, *Good* → prob(Good) élevée.
- If there's a large overlap, the model confuses classes.
"""
)

st.subheader("Logistic Regression Model Performance")
col3, col4 = st.columns(2)
with col3:
    try:
        ll_value = log_loss(y_test, y_proba)
        st.write(f"**Log Loss:** {ll_value:.3f} (Lower is better)")
    except ValueError:
        st.write("⚠️ Log Loss not available (single class in test?).")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    st.write(
        f"**Balanced Accuracy:** {bal_acc:.3f} (average recall across classes, better for class imbalance.)"
    )
with col4:
    r2_value = r2_score(y_test, y_pred)
    st.write(f"**R² Score:** {r2_value:.3f} (not very meaningful for classification)")

st.divider()

# XGBoost Classifier
st.subheader("XGBoost Classifier - Quality")

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["quality", "quality_cat"], axis=1),
    df["quality"],
    test_size=0.2,
    random_state=42,
    stratify=df["quality"],
)

class_mapping = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}
y_train_mapped = y_train.replace(class_mapping)
y_test_mapped = y_test.replace(class_mapping)

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

st.subheader("Confusion Matrix (XGBoost)")
col5, col6 = st.columns(2)
with col5:
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_original, y_pred_original, output_dict=True)
        ).transpose()
    )
with col6:
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

st.divider()

st.subheader("📊 Random Forest - Quality")
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["quality", "quality_cat"], axis=1),
    df["quality"],
    test_size=0.2,
    random_state=42,
    stratify=df["quality"],
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
y_pred_rf = rf_model.predict(X_test)
y_test_str = y_test.replace(quality_labels)
y_pred_str = pd.Series(y_pred_rf).replace(quality_labels)

st.subheader("Confusion Matrix")
col7, col8 = st.columns(2)
with col7:
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(
                y_test_str, y_pred_str, output_dict=True, zero_division=1
            )
        ).transpose()
    )
with col8:
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
st.write(
    f"**Brier Score:** {np.mean(brier_scores):.3f} (mean squared error of predicted probabilities vs actual outcome (lower=better)."
)

st.divider()

st.subheader("📊 Random Forest - Quality Category")
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
st.subheader("Confusion Matrix")
col9, col10 = st.columns(2)
with col9:
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_str, y_pred_str, output_dict=True)
        ).transpose()
    )
with col10:
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

st.divider()

st.subheader("📊 Random Forest - Fixed Acidity")
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
st.subheader("Confusion Matrix")
col11, col12 = st.columns(2)
with col11:
    st.write("**Classification Report:**")
    st.write(
        pd.DataFrame(
            classification_report(y_test_str, y_pred_str, output_dict=True)
        ).transpose()
    )
with col12:
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

st.divider()

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
