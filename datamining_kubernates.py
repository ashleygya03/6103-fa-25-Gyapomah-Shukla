# %% [markdown]
# DATS-6103 Data Mining – Final Project
# Title: Predicting Kubernetes Pod Overload Using Machine Learning
# Authors: Aditi Shukla, Ashley Gyapomah
# 
# To run:
# - Install: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost (optional).
# - Place `kubernetes_resource_allocation_dataset.csv` in the same folder

# %%

#  LOADING & CLEANING DATA


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

#Loading dataset
df = pd.read_csv("kubernetes_resource_allocation_dataset.csv")

# Dataset source:
# Kubernetes resource allocation dataset (link: https://www.kaggle.com/datasets/nickkinyae/kubernetes-resource-and-performancemetricsallocation/data )


print("Original shape:", df.shape)
display(df.head())

#Standardizing column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nCleaned column names:")
print(df.columns.tolist())

#Convert timestamp if exists
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("\nINFO AFTER CLEANING:")
print(df.info())

print("\nMissing values per column:")
print(df.isna().sum())


# %%
df.info()

# %%
df.describe()

# %%
df.columns.tolist()

# %%
df.mean(numeric_only=True)

# %%
df.var(numeric_only=True)

# %%

# DATA PREPROCESSING


# Creating a working copy to avoid modifying the original
df_clean = df.copy()

# Core numeric columns we rely on
key_cols = [
    "cpu_request", "cpu_limit", "memory_request", "memory_limit",
    "cpu_usage", "memory_usage"
]

# Removing rows where these essential columns have missing values
df_clean = df_clean.dropna(subset=key_cols)

# Removing rows with negative values (invalid readings)
for col in key_cols:
    df_clean = df_clean[df_clean[col] >= 0]

print("Shape after core cleaning:", df_clean.shape)

# displaying cleaned data
display(df_clean[key_cols].head())


# %%

#FEATURE ENGINEERING

df_fe = df_clean.copy()  # work on a fresh copy
eps = 1e-6  # small number to avoid division by zero

#Utilization Features
df_fe["cpu_utilization"] = df_fe["cpu_usage"] / (df_fe["cpu_limit"] + eps)
df_fe["memory_utilization"] = df_fe["memory_usage"] / (df_fe["memory_limit"] + eps)

# Usage vs Request Ratios 
df_fe["cpu_request_ratio"] = df_fe["cpu_usage"] / (df_fe["cpu_request"] + eps)
df_fe["memory_request_ratio"] = df_fe["memory_usage"] / (df_fe["memory_request"] + eps)

# Combined Load Metric
df_fe["overall_load"] = (df_fe["cpu_utilization"] + df_fe["memory_utilization"]) / 2

# Displaying engineered columns
print("Engineered feature columns:")
display(df_fe[[
    "cpu_utilization",
    "memory_utilization",
    "cpu_request_ratio",
    "memory_request_ratio",
    "overall_load"
]].head())


# %%

# EXPLORATORY DATA ANALYSIS (EDA)


plt.style.use("seaborn-v0_8")

# 1. Distribution of CPU Utilization
plt.figure(figsize=(6,4))
df_fe["cpu_utilization"].hist(bins=50, color="steelblue")
plt.title("CPU Utilization Distribution")
plt.xlabel("CPU Utilization")
plt.ylabel("Frequency")
plt.show()

# 2. Distribution of Memory Utilization
plt.figure(figsize=(6,4))
df_fe["memory_utilization"].hist(bins=50, color="darkgreen")
plt.title("Memory Utilization Distribution")
plt.xlabel("Memory Utilization")
plt.ylabel("Frequency")
plt.show()

# 3. Joint relationship: CPU vs Memory Utilization 
plt.figure(figsize=(6,4))
plt.scatter(df_fe["cpu_utilization"], df_fe["memory_utilization"], alpha=0.3, s=10)
plt.title("CPU vs Memory Utilization")
plt.xlabel("CPU Utilization")
plt.ylabel("Memory Utilization")
plt.show()

# 4. Correlation Heatmap of Numeric Features
numeric_cols = df_fe.select_dtypes(include=[np.number]).columns
corr = df_fe[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()


# %%

# PART 5 — STATISTICAL TESTING

from scipy.stats import ttest_ind, f_oneway, chi2_contingency

#  T-TEST 
# Comparing CPU usage for overloaded vs non-overloaded (after we create the target later)
# For now, let's define a temporary overload threshold
cpu_thresh = 0.75
df_fe["overloaded_temp"] = (df_fe["cpu_utilization"] > cpu_thresh).astype(int)

group1 = df_fe[df_fe["overloaded_temp"] == 1]["cpu_usage"]
group0 = df_fe[df_fe["overloaded_temp"] == 0]["cpu_usage"]

t_stat, p_val = ttest_ind(group1, group0, equal_var=False)

print("\n T-TEST: CPU Usage vs Overload ")
print("T-statistic:", round(t_stat, 4))
print("P-value:", p_val)
print("Interpretation:",
      "Significant difference" if p_val < 0.05 else "No significant difference")


# ANOVA 
print("\nANOVA: CPU Usage Across Namespaces")
groups = [group["cpu_usage"].values for _, group in df_fe.groupby("namespace")]
anova_stat, anova_p = f_oneway(*groups)

print("ANOVA F-statistic:", round(anova_stat, 4))
print("P-value:", anova_p)
print("Interpretation:",
      "Groups differ significantly" if anova_p < 0.05 else "No significant difference")


#CHI-SQUARE 
print("\n CHI-SQUARE: Scaling Policy vs Overload")
contingency = pd.crosstab(df_fe["scaling_policy"], df_fe["overloaded_temp"])
chi2, p, _, _ = chi2_contingency(contingency)

print("Chi-square:", round(chi2, 4))
print("P-value:", p)
print("Interpretation:",
      "Variables are dependent" if p < 0.05 else "Variables are independent")


# %%

# PART 6 — DEFINE TARGET VARIABLE


df_model = df_fe.copy()

# Overload thresholds
cpu_thresh = 0.75
mem_thresh = 0.75

# Logical overload condition
df_model["overloaded_now"] = (
    (df_model["cpu_utilization"] > cpu_thresh) |
    (df_model["memory_utilization"] > mem_thresh)
).astype(int)

# Final target
df_model["need_new_pod"] = df_model["overloaded_now"]

print("Target distribution (need_new_pod):")
print(df_model["need_new_pod"].value_counts())
print("Overload rate:", round(df_model["need_new_pod"].mean(), 4))

# Displaying the final dataset with target
display(df_model[[
    "cpu_utilization", "memory_utilization",
    "overall_load", "need_new_pod"
]].head())


# %%

#  TRAIN-TEST SPLIT + FEATURE SELECTION


df_ml = df_model.copy()

#Feature Matrix & Target
X = df_ml.drop(columns=["need_new_pod"])  # all columns except target
y = df_ml["need_new_pod"]

# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class balance consistent
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

#Identify Feature Types
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# %%

# PREPROCESSING PIPELINE

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numeric transformer
numeric_transformer = StandardScaler()

# Categorical transformer
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combined preprocessing pipeline 
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully.")


# %%

#  MODEL TRAINING


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
    print("XGBoost not installed — skipping.")


#Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )


# Train and Evaluate Each Model
results = []

for name, clf in models.items():
    print(f"\nTraining {name} ")

    # Building full pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc = float("nan")
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("ROC-AUC  :", round(roc, 4))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Store results
    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    })


# Comparison Table
import pandas as pd
results_df = pd.DataFrame(results)
print("\n Model Performance Comparison")
display(results_df.sort_values(by="f1", ascending=False))


# %%

# VISUAL MODEL COMPARISON


plt.figure(figsize=(10,5))

plt.bar(results_df["model"], results_df["f1"], color="skyblue")
plt.title("Model Comparison — F1 Score")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.ylim(0.9, 1.01)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()


plt.figure(figsize=(10,5))
plt.bar(results_df["model"], results_df["roc_auc"], color="lightgreen")
plt.title("Model Comparison — ROC AUC")
plt.ylabel("ROC AUC Score")
plt.xticks(rotation=45)
plt.ylim(0.95, 1.01)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()



