import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="centered")
st.title("Analyse and Machine Learning on Red Wine Quality dataset")

df = pd.read_csv('data/winequality-red.csv', sep=';')


## Data Preprocessing
def categorize_quality(q):
    if q <= 5:
        return -1
    elif q == 6:
        return 0
    else:
        return 1


def categorize_acidity(acidity):
    if acidity < 8.0:
        return -1
    elif acidity < 11.0:
        return 0
    else:
        return 1


df = df.astype(float)

# Categorize quality and acidity
df["quality_cat"] = df["quality"].apply(categorize_quality)
df["fixed_acidity_cat"] = df["fixed acidity"].apply(categorize_acidity)

# Mapping of quality and acidity categories in a dictionary
quality_labels = {-1: "Bad", 0: "Average", 1: "Good"}
acidity_labels = {-1: "Low Acidity", 0: "Medium Acidity", 1: "High Acidity"}

## Data Overview
st.subheader("Overview of the dataset")
st.write(df.head())
st.subheader("Number of data (rows, columns)")
st.write(df.shape)

# Exploration of the columns
st.subheader("Exploration of the columns")
for col in df.columns:
    unique_values = df[col].unique()
    st.write("--------------------", col, "--------------------")
    st.write(f"🔹 number of unique value {col} : {len(unique_values)}")
    st.write(f"🔹 number of null value {col} : {df[col].isnull().sum()}")
    st.write(f"🔹 min value for {col} : {df[col].min()}")
    st.write(f"🔹 max value for {col} : {df[col].max()}")

# Explanation of categories
st.subheader("🔹 Explanation of categories")
st.write("We have created categories for the quality and fixed acidity of the wines, to facilitate the analysis.")

st.markdown("""
### **Wine Quality (`quality_cat`)**:
- 🟥 **Bad** (`≤ 5`)
- 🟨 **Average** (`= 6`)
- 🟩 **Good** (`≥ 7`)

### **Fixed Acidity (`fixed_acidity_cat`)**:
- 🔵 **Low Acidity** (`< 8.0`)
- 🟠 **Medium Acidity** (`8.0 ≤ acidity < 11.0`)
- 🔴 **High Acidity** (`≥ 11.0`)
""")


## Data visualization

# Quality distribution
st.markdown("---")
st.subheader("📊 Data Visualization")
fig, ax = plt.subplots()
ax.hist(df["quality"], bins=np.arange(df["quality"].min(), df["quality"].max() + 1, 1), color="red", alpha=0.7,
        edgecolor="black")
ax.set_title("Quality distribution")
ax.set_xlabel("Quality")
ax.set_ylabel("Number of wines")
st.pyplot(fig)

# Correlation matrix
df_corr = df.drop(["quality_cat", "fixed_acidity_cat"], axis=1)

st.markdown("---")
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
st.markdown("---")
st.subheader("Characteristics boxplot of the dataset")
selected_col = st.selectbox("Select a column", df.columns[:-2])
fig, ax = plt.subplots()
sns.boxplot(y=df[selected_col], ax=ax, color="lightblue", showfliers=True)
ax.set_title(f"Boxplot of {selected_col}")
st.pyplot(fig)

# Boxplot by categories
st.subheader("Boxplot by categories")
col_to_plot = st.radio("Select a category to compare", ["quality_cat", "fixed_acidity_cat"])
continuous_feature = st.selectbox("Select a continuous variable for comparison", ["alcohol", "citric acid", "density", "pH"])
df_display = df.copy()
df_display["quality_cat"] = df_display["quality_cat"].replace(quality_labels)
df_display["fixed_acidity_cat"] = df_display["fixed_acidity_cat"].replace(acidity_labels)

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=df_display[col_to_plot], y=df_display[continuous_feature], palette="coolwarm", ax=ax)
ax.set_xlabel(col_to_plot.replace("_", " ").title())
ax.set_ylabel(continuous_feature.replace("_", " ").title())
ax.set_title(f"Boxplot of {continuous_feature.replace('_', ' ').title()} by {col_to_plot.replace('_', ' ').title()}")
st.pyplot(fig)

## Machine learning
st.markdown("---")
st.subheader("🤖 Machine Learning - Random Forest Models")

# Quality classification
st.subheader("📊 Random Forest - Wine Quality")
X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality', 'quality_cat'], axis=1), df['quality'],
                                                    test_size=0.2, random_state=42, stratify=df['quality'])
y_train = y_train.replace({3: 4})
y_test = y_test.replace({3: 4})

# print(pd.Series(y_train).value_counts())

rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=3, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42)

rf_model.fit(X_train, y_train)

# Feature importance
importances = rf_model.feature_importances_
features = X_train.columns
fig, ax = plt.subplots(figsize=(10, 6))
pd.Series(importances, index=features).sort_values(ascending=False).plot(kind="bar", ax=ax)
ax.set_title("Importance of Features")
st.pyplot(fig)

# Model performance
y_pred = rf_model.predict(X_test)
y_test_str = y_test.replace(quality_labels)
y_pred_str = pd.Series(y_pred).replace(quality_labels)

st.markdown("---")
st.subheader("📊 Model Performance for randomForest about quality")
st.write("**Classification Report:**")
st.write(pd.DataFrame(classification_report(y_test_str, y_pred_str, output_dict=True)).transpose())
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
    mask = (y_test == i)
    if mask.sum() > 0:
        brier_scores.append(brier_score_loss(mask, y_proba[:, i]))
st.write(f"**Brier Score:** {np.mean(brier_scores):.3f}")

# Confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)


## Quality category classification

st.markdown("---")
st.subheader("📊 Random Forest - Quality Category")
X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality', 'quality_cat'], axis=1), df['quality_cat'],
                                                    test_size=0.2, random_state=42, stratify=df['quality_cat'])
rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=3, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Model performance
y_pred = rf_model.predict(X_test)
y_test_str = y_test.replace(quality_labels)
y_pred_str = pd.Series(y_pred).replace(quality_labels)
st.subheader("📊 Model Performance for randomForest about quality_cat")
st.write("**Classification Report:**")
st.write(pd.DataFrame(classification_report(y_test_str, y_pred_str, output_dict=True)).transpose())
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
    mask = (y_test == i)
    if mask.sum() > 0:
        brier_scores.append(brier_score_loss(mask, y_proba[:, i]))
st.write(f"**Brier Score:** {np.mean(brier_scores):.3f}")

# Confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test_str, y_pred_str), annot=True, cmap="Blues", fmt="d",
            xticklabels=sorted(quality_labels.values()),
            yticklabels=sorted(quality_labels.values()))
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)



## Fixed acidity classification

st.markdown("---")
st.subheader("📊 Random Forest - Fixed Acidity")
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['fixed acidity', 'fixed_acidity_cat'], axis=1),
    df['fixed_acidity_cat'],
    test_size=0.2,
    random_state=42,
    stratify=df['fixed_acidity_cat']
)

rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=3, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_test_str = y_test.replace(acidity_labels)
y_pred_str = pd.Series(y_pred).replace(acidity_labels)
st.subheader("📊 Model Performance for randomForest about fixed acidity")
st.write("**Classification Report:**")
st.write(pd.DataFrame(classification_report(y_test_str, y_pred_str, output_dict=True)).transpose())
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
    mask = (y_test == i)
    if mask.sum() > 0:
        brier_scores.append(brier_score_loss(mask, y_proba[:, i]))
st.write(f"**Brier Score:** {np.mean(brier_scores):.3f}")

# Confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test_str, y_pred_str), annot=True, cmap="Blues", fmt="d",
            xticklabels=sorted(acidity_labels.values()),
            yticklabels=sorted(acidity_labels.values()))
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)


# Clustering with KMeans

st.markdown("---")
st.subheader("Clustering with KMeans")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('quality', axis=1))

# Determine optimal number of clusters using Elbow method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(k_values, inertia, marker='o')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal K')
st.pyplot(fig)
st.write("The optimal number of clusters is hard to see, we will choose k=5")

# Apply K-Means with an optimal number of clusters (e.g., k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Scatter plot of first two principal components
st.subheader("Cluster Visualization (Alcool & Quality)")
fig, ax = plt.subplots()
sns.scatterplot(x=df.iloc[:, 10], y=df.iloc[:, 11], hue=df['cluster'], palette='viridis', ax=ax)
ax.set_xlabel(df.columns[10])
ax.set_ylabel(df.columns[11])
ax.set_title("Clusters based on 11th and 12th Features")
st.pyplot(fig)

# Pairplot of clustered data
st.subheader("Pairplot of Clustered Data")
st.write("Pairplot of selected features colored by cluster")
selected_features = st.multiselect("Select up to 4 features for visualization", df.columns[:-1], default=df.columns[:4])
if selected_features:
    pairplot_fig = sns.pairplot(df, vars=selected_features, hue='cluster', palette='viridis')
    st.pyplot(pairplot_fig)

### use xgboost
