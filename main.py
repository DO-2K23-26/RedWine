import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="centered")
st.title("Analyse and Machine Learning on Red Wine Quality dataset")

df = pd.read_csv('data/winequality-red.csv', sep=';')
df = df.astype(float)
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

#X_train, X_test, y_train, y_test = train_test_split(df[['alcohol', 'sulphates', 'citric acid']], df['quality'], test_size=0.2, random_state=42, stratify=df['quality'])

X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality'], axis=1), df['quality'], test_size=0.2, random_state=42, stratify=df['quality'])
y_train = y_train.replace({3: 4})
y_test = y_test.replace({3: 4})

print(pd.Series(y_train).value_counts())

rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=3, min_samples_leaf=1, class_weight='balanced', random_state=42)


rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
features = X_train.columns

fig, ax = plt.subplots(figsize=(10,6))
pd.Series(importances, index=features).sort_values(ascending=False).plot(kind="bar", ax=ax)
ax.set_title("Importance des Features")

# Afficher dans Streamlit
st.pyplot(fig)

y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

print("Balanced Accuracy :", balanced_accuracy_score(y_test, y_pred))
# Meilleurs paramètres : {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None}
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

print("Scores de validation croisée :", cv_scores)
print("Score moyen :", cv_scores.mean())
print("Score du modèle", rf_model.score(X_test, y_test))


# Clustering with KMeans

# Clustering with KMeans
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
