import matplotlib
matplotlib.use("Agg")
import pandas as pd

# Task 1: Load the dataset
df = pd.read_csv("WineQT.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display last 5 rows
print("\nLast 5 Rows:")
print(df.tail())

# Display 5 random rows
print("\nRandom 5 Rows:")
print(df.sample(5, random_state=42))

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display target column
print("\nTarget Column: quality")

# Display all column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display feature meanings
print("\nFeature Meanings:")
features = {
    "fixed acidity": "Non-volatile acids present in wine",
    "volatile acidity": "Acetic acid level; high values may cause an unpleasant taste",
    "citric acid": "Citric acid content",
    "residual sugar": "Sugar remaining after fermentation",
    "chlorides": "Salt content",
    "free sulfur dioxide": "Free sulfur dioxide content",
    "total sulfur dioxide": "Total sulfur dioxide content",
    "density": "Density of wine",
    "pH": "Acidity or alkalinity level",
    "sulphates": "Sulphate content",
    "alcohol": "Alcohol percentage",
    "quality": "Wine quality score (target variable)",
    "Id": "Row identifier, not a wine-quality feature"
}

for feature, meaning in features.items():
    if feature in df.columns:
        print(f"{feature}: {meaning}")


# Task 2: Basic Data Inspection

print("\n========== TASK 2: BASIC DATA INSPECTION ==========")

# 1. Dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# 2. Column names
print("\n2. Column Names:")
print(df.columns.tolist())

# 3. Data types
print("\n3. Data Types:")
print(df.dtypes)

# 4. Unique values in each column
print("\n4. Unique Values:")
print(df.nunique())

# 5. Wine quality distribution
print("\n5. Quality Distribution:")
print(df["quality"].value_counts().sort_index())

# 6. Statistical summary
print("\n6. Statistical Summary:")
print(df.describe())

# 7. Missing values
print("\n7. Missing Values:")
print(df.isnull().sum())

# 8. Duplicate rows
print("\n8. Duplicate Rows:")
print(df.duplicated().sum())


# Task 3: Data Cleaning

print("\n========== TASK 3: DATA CLEANING ==========")

# 1. Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 2. Remove duplicate rows
duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows Before Cleaning:", duplicates_before)

df = df.drop_duplicates()

print("Duplicate Rows After Cleaning:", df.duplicated().sum())

# 3. Remove Id column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    print("\nId column removed.")
else:
    print("\nId column not found.")

# 4. Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 5. Display final dataset shape
print("\nDataset Shape After Cleaning:")
print(df.shape)

# 6. Display cleaned dataset
print("\nCleaned Dataset Preview:")
print(df.head())


# Task 4: Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

print("\n========== TASK 4: EXPLORATORY DATA ANALYSIS ==========")

# 1. Wine quality distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="quality")
plt.title("Wine Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Number of Wines")
plt.tight_layout()
plt.show()

# 2. Alcohol distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="alcohol", bins=20, kde=True)
plt.title("Alcohol Distribution")
plt.xlabel("Alcohol Percentage")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Volatile acidity distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="volatile acidity", bins=20, kde=True)
plt.title("Volatile Acidity Distribution")
plt.xlabel("Volatile Acidity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Alcohol vs. quality
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="quality", y="alcohol")
plt.title("Alcohol Content vs. Wine Quality")
plt.xlabel("Quality Score")
plt.ylabel("Alcohol Percentage")
plt.tight_layout()
plt.show()

print("Task 4 EDA completed successfully.")


# Task 5: Correlation Analysis

print("\n========== TASK 5: CORRELATION ANALYSIS ==========")

# Calculate correlation between numerical columns
correlation_matrix = df.corr(numeric_only=True)

print("\nCorrelation with Wine Quality:")
print(correlation_matrix["quality"].sort_values(ascending=False))

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Wine Quality Dataset")
plt.tight_layout()
plt.show()

print("\nTask 5 completed successfully.")


# Task 6: Feature and Target Separation

print("\n========== TASK 6: FEATURE AND TARGET SEPARATION ==========")

# Separate features (X) and target (y)
X = df.drop(columns=["quality", "Id"], errors="ignore")
y = df["quality"]

# Display feature and target shapes
print("\nFeatures Shape (X):", X.shape)
print("Target Shape (y):", y.shape)

# Display feature names
print("\nFeature Columns:")
print(X.columns.tolist())

# Display target values
print("\nTarget Column: quality")
print(y.head())

# Check target class distribution
print("\nTarget Class Distribution:")
print(y.value_counts().sort_index())

print("\nTask 6 completed successfully.")


# Task 7: Train-Test Split

from sklearn.model_selection import train_test_split

print("\n========== TASK 7: TRAIN-TEST SPLIT ==========")

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display shapes
print("\nOriginal Features:", X.shape)
print("Original Target:", y.shape)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target:", y_train.shape)
print("Testing Target:", y_test.shape)

print("\nTask 7 completed successfully.")


# Task 8: Feature Scaling

from sklearn.preprocessing import StandardScaler

print("\n========== TASK 8: FEATURE SCALING ==========")

# Create the scaler
scaler = StandardScaler()

# Fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Display shapes
print("\nOriginal Training Features Shape:", X_train.shape)
print("Scaled Training Features Shape:", X_train_scaled.shape)

print("\nOriginal Testing Features Shape:", X_test.shape)
print("Scaled Testing Features Shape:", X_test_scaled.shape)

# Display a few scaled rows
print("\nFirst 5 Rows of Scaled Training Data:")
print(X_train_scaled[:5])

print("\nTask 8 completed successfully.")


# Task 9: Train Machine Learning Models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("\n========== TASK 9: MODEL TRAINING ==========")

# Create classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train each model
trained_models = {}

for name, model in models.items():
    if name == "Random Forest":
        # Tree-based models do not require feature scaling
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_scaled, y_train)

    trained_models[name] = model
    print(f"{name} trained successfully!")

print("\nAll models trained successfully.")


# Task 10: Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

print("\n========== TASK 10: MODEL EVALUATION ==========")

results = []

for name, model in trained_models.items():

    # Random Forest uses unscaled test data
    if name == "Random Forest":
        predictions = model.predict(X_test)
    else:
        predictions = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    })

# Display comparison table
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

print("\nTask 10 completed successfully.")


# TASK 11: CONFUSION MATRIX

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\n========== TASK 11: CONFUSION MATRIX ==========")

# Use the trained Random Forest model
best_model = trained_models["Random Forest"]
predictions = best_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

print("Task 11 completed successfully.")


# Task 12: Identify the Best Model

print("\nTask 12: Best Model Selection")

best_model = results_df.sort_values(
    by="F1-Score",
    ascending=False
).iloc[0]

print("Best Model:", best_model["Model"])
print("Accuracy:", best_model["Accuracy"])
print("F1 Score:", best_model["F1-Score"])

