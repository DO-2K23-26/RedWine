import pandas as pd

# Load data
df = pd.read_csv("data/winequality-red.csv", sep=";")


# Data Preprocessing
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


# Categorize quality
df = df.astype(float)
# Categorize quality and acidity
df["quality_cat"] = df["quality"].apply(categorize_quality)
df["fixed_acidity_cat"] = df["fixed acidity"].apply(categorize_acidity)

# Mapping of quality and acidity categories in a dictionary
quality_labels = {-1: "Bad", 0: "Average", 1: "Good"}
acidity_labels = {-1: "Low Acidity", 0: "Medium Acidity", 1: "High Acidity"}
