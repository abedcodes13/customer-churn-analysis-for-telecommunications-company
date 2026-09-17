# ============================================
# TYLER - DATA ENGINEER
# Stage 2 - Data Preparation & Feature Engineering
# Customer Churn Analysis
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("Dataset_ATS_v2.csv")

print("============================================")
print("1. LOAD DATASET")
print("============================================")

print("Original dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================
# 2. Initial Data Quality Check
# ============================================

print("\n============================================")
print("2. INITIAL DATA QUALITY CHECK")
print("============================================")

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)

print("\n--- Duplicate Rows ---")
print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ============================================
# 3. Remove Exact Duplicate Rows
# ============================================

print("\n============================================")
print("3. REMOVE EXACT DUPLICATE ROWS")
print("============================================")

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)

print(
    "Removed duplicate rows:",
    duplicate_count
)

print(
    "Dataset shape after removing duplicates:",
    df.shape
)


# ============================================
# 4. Define Columns
# ============================================

print("\n============================================")
print("4. DEFINE COLUMNS")
print("============================================")

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges"
]

categorical_columns = [
    "gender",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "Contract"
]

target_column = "Churn"

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nTarget column:")
print(target_column)


# ============================================
# 5. Check Required Columns
# ============================================

print("\n============================================")
print("5. CHECK REQUIRED COLUMNS")
print("============================================")

required_columns = (
    numerical_columns
    + categorical_columns
    + [target_column]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# ============================================
# 6. Handle Missing Values
# ============================================

print("\n============================================")
print("6. HANDLE MISSING VALUES")
print("============================================")

print("\nMissing values before handling:")
print(df.isnull().sum())

# Numerical columns:
# Use median if missing values exist.
for column in numerical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )

        print(
            f"Filled missing values in {column} "
            f"using median."
        )


# Categorical columns:
# Use mode if missing values exist.
for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

        print(
            f"Filled missing values in {column} "
            f"using mode."
        )


# Target variable:
# Remove rows if Churn is missing.
if df[target_column].isnull().sum() > 0:

    rows_before = len(df)

    df = df.dropna(
        subset=[target_column]
    ).reset_index(drop=True)

    rows_removed = rows_before - len(df)

    print(
        f"Removed {rows_removed} rows "
        f"with missing Churn values."
    )


print("\nMissing values after handling:")
print(df.isnull().sum())

print(
    "\nTotal missing values after handling:",
    df.isnull().sum().sum()
)


# ============================================
# 7. FEATURE ENGINEERING (NEW)
# ============================================
# Task from Week 4 meeting: "complete feature
# identification and engineering" (Tyler's action
# item). The dataset only has 10 raw fields (no
# billing history / payment method per Arafat's
# note), so new features are derived from the
# existing tenure / MonthlyCharges / service columns
# to give Abed (clustering) and Felix (ANN) more
# signal to work with.

print("\n============================================")
print("7. FEATURE ENGINEERING")
print("============================================")

# 7a. TenureGroup - bucket tenure into lifecycle stages.
# Useful for clustering interpretation and as a
# categorical signal for the ANN.
tenure_bins = [-1, 12, 24, 48, np.inf]
tenure_labels = ["New", "Established", "Loyal", "Veteran"]

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels
)

print("\nTenureGroup distribution:")
print(df["TenureGroup"].value_counts())

# 7b. EstimatedTotalSpend - proxy for TotalCharges,
# which is not present in this dataset.
df["EstimatedTotalSpend"] = (
    df["tenure"] * df["MonthlyCharges"]
)

print(
    "\nEstimatedTotalSpend (tenure x MonthlyCharges) "
    "created."
)

# 7c. NumActiveServices - count of active services
# from PhoneService, MultipleLines, InternetService.
def count_active_services(row):
    count = 0

    if row["PhoneService"] == "Yes":
        count += 1

    if row["MultipleLines"] == "Yes":
        count += 1

    if row["InternetService"] != "No":
        count += 1

    return count

df["NumActiveServices"] = df.apply(
    count_active_services,
    axis=1
)

print("\nNumActiveServices distribution:")
print(df["NumActiveServices"].value_counts())

# 7d. AvgMonthlyChargePerService - spend efficiency
# per active service (+1 to avoid division by zero
# for customers with no active services).
df["AvgMonthlyChargePerService"] = (
    df["MonthlyCharges"] / (df["NumActiveServices"] + 1)
)

print(
    "\nAvgMonthlyChargePerService "
    "(MonthlyCharges / (NumActiveServices + 1)) created."
)

# Add the new engineered features to the column lists
# so they flow through encoding / scaling below.
numerical_columns = numerical_columns + [
    "EstimatedTotalSpend",
    "NumActiveServices",
    "AvgMonthlyChargePerService"
]

categorical_columns = categorical_columns + [
    "TenureGroup"
]

print("\nUpdated numerical columns:")
print(numerical_columns)

print("\nUpdated categorical columns:")
print(categorical_columns)

print(
    "\nDataset shape after feature engineering:",
    df.shape
)


# ============================================
# 8. Check Categorical Values
# ============================================

print("\n============================================")
print("8. CATEGORICAL VALUE CHECK")
print("============================================")

for column in categorical_columns:

    print(f"\n--- {column} ---")

    print(
        df[column].value_counts()
    )


# ============================================
# 9. Separate Features and Target
# ============================================

print("\n============================================")
print("9. SEPARATE FEATURES AND TARGET")
print("============================================")

X = df.drop(
    target_column,
    axis=1
)

y = df[target_column]

print(
    "Feature dataset shape:",
    X.shape
)

print(
    "Target dataset shape:",
    y.shape
)


# ============================================
# 10. Encode Categorical Features
# ============================================

print("\n============================================")
print("10. ENCODE CATEGORICAL FEATURES")
print("============================================")

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print(
    "Shape after categorical encoding:",
    X.shape
)

print("\nEncoded feature columns:")

for column in X.columns:
    print(column)


# ============================================
# 11. Encode Target Variable
# ============================================

print("\n============================================")
print("11. ENCODE TARGET VARIABLE")
print("============================================")

y = y.map({
    "No": 0,
    "Yes": 1
})

# Check whether target encoding created
# any missing values.
if y.isnull().sum() > 0:

    raise ValueError(
        "Unexpected values found in Churn column."
    )

print("\nEncoded Churn distribution:")
print(y.value_counts())

print("\nChurn percentage:")
print(
    (y.value_counts(normalize=True) * 100)
)


# ============================================
# 12. Create Complete Preprocessed Dataset
# ============================================

print("\n============================================")
print("12. CREATE PREPROCESSED DATASET")
print("============================================")

preprocessed_df = X.copy()

preprocessed_df[target_column] = y.values

print(
    "Preprocessed dataset shape:",
    preprocessed_df.shape
)

print("\nFirst 5 rows:")
print(
    preprocessed_df.head()
)


# ============================================
# 13. Save Preprocessed Dataset
# ============================================

print("\n============================================")
print("13. SAVE PREPROCESSED DATASET")
print("============================================")

preprocessed_df.to_csv(
    "ATS_preprocessed_dataset.csv",
    index=False
)

print(
    "Saved: ATS_preprocessed_dataset.csv"
)


# ============================================
# 14. Train / Test Split
# ============================================

print("\n============================================")
print("14. TRAIN / TEST SPLIT")
print("============================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training feature shape:",
    X_train.shape
)

print(
    "Testing feature shape:",
    X_test.shape
)

print(
    "Training target shape:",
    y_train.shape
)

print(
    "Testing target shape:",
    y_test.shape
)

print(
    "\nTraining percentage:",
    round(
        len(X_train) / len(X) * 100,
        2
    ),
    "%"
)

print(
    "Testing percentage:",
    round(
        len(X_test) / len(X) * 100,
        2
    ),
    "%"
)


# ============================================
# 15. Check Churn Distribution
# ============================================

print("\n============================================")
print("15. TRAIN / TEST CHURN DISTRIBUTION")
print("============================================")

print("\nTraining Churn distribution:")
print(y_train.value_counts())

print("\nTesting Churn distribution:")
print(y_test.value_counts())


# ============================================
# 16. Feature Scaling
# ============================================

print("\n============================================")
print("16. FEATURE SCALING")
print("============================================")

scaler = StandardScaler()

# IMPORTANT:
# Fit the scaler ONLY on training data.
# This now also covers the newly engineered numeric
# features (EstimatedTotalSpend, NumActiveServices,
# AvgMonthlyChargePerService) since they were added
# to X before the split above.
X_train_scaled = scaler.fit_transform(
    X_train
)

# Use the same scaler on testing data.
X_test_scaled = scaler.transform(
    X_test
)

print(
    "Scaling technique: StandardScaler"
)

print(
    "\nTraining data:"
)

print(
    "scaler.fit_transform(X_train)"
)

print(
    "\nTesting data:"
)

print(
    "scaler.transform(X_test)"
)


# ============================================
# 17. Convert Scaled Data to DataFrames
# ============================================

print("\n============================================")
print("17. CREATE SCALED DATAFRAMES")
print("============================================")

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

print(
    "Scaled training data shape:",
    X_train_scaled.shape
)

print(
    "Scaled testing data shape:",
    X_test_scaled.shape
)


# ============================================
# 18. Final Data Quality Check
# ============================================

print("\n============================================")
print("18. FINAL DATA QUALITY CHECK")
print("============================================")

print("\n--- Preprocessed Dataset ---")

print(
    "Rows:",
    preprocessed_df.shape[0]
)

print(
    "Columns:",
    preprocessed_df.shape[1]
)

print("\n--- Training Dataset ---")

print(
    "Rows:",
    X_train_scaled.shape[0]
)

print(
    "Columns:",
    X_train_scaled.shape[1]
)

print("\n--- Testing Dataset ---")

print(
    "Rows:",
    X_test_scaled.shape[0]
)

print(
    "Columns:",
    X_test_scaled.shape[1]
)


print("\n--- Missing Values ---")

print(
    "Preprocessed dataset:",
    preprocessed_df.isnull().sum().sum()
)

print(
    "Training dataset:",
    X_train_scaled.isnull().sum().sum()
)

print(
    "Testing dataset:",
    X_test_scaled.isnull().sum().sum()
)


print("\n--- Duplicate Rows ---")

print(
    "Preprocessed dataset:",
    preprocessed_df.duplicated().sum()
)

print(
    "Training dataset:",
    X_train_scaled.duplicated().sum()
)

print(
    "Testing dataset:",
    X_test_scaled.duplicated().sum()
)


# ============================================
# 19. Display Prepared Training Data
# ============================================

print("\n============================================")
print("19. PREPARED TRAINING DATA")
print("============================================")

print(
    X_train_scaled.head()
)


# ============================================
# 20. Save Training and Testing Datasets
# ============================================

print("\n============================================")
print("20. SAVE TRAINING AND TESTING DATA")
print("============================================")

X_train_scaled.to_csv(
    "ATS_X_train_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    "ATS_X_test_scaled.csv",
    index=False
)

y_train.to_csv(
    "ATS_y_train.csv",
    index=False
)

y_test.to_csv(
    "ATS_y_test.csv",
    index=False
)

print(
    "Saved: ATS_X_train_scaled.csv"
)

print(
    "Saved: ATS_X_test_scaled.csv"
)

print(
    "Saved: ATS_y_train.csv"
)

print(
    "Saved: ATS_y_test.csv"
)


# ============================================
# 21. Save StandardScaler
# ============================================

print("\n============================================")
print("21. SAVE STANDARD SCALER")
print("============================================")

joblib.dump(
    scaler,
    "standard_scaler.pkl"
)

print(
    "Saved: standard_scaler.pkl"
)


# ============================================
# 22. Final Summary
# ============================================

print("\n============================================")
print("22. DATA PREPARATION SUMMARY")
print("============================================")

print(
    "Original dataset after cleaning:",
    df.shape
)

print(
    "Preprocessed dataset:",
    preprocessed_df.shape
)

print(
    "Training features:",
    X_train_scaled.shape
)

print(
    "Testing features:",
    X_test_scaled.shape
)

print(
    "Training target:",
    y_train.shape
)

print(
    "Testing target:",
    y_test.shape
)

print(
    "\nEngineered features added: "
    "TenureGroup, EstimatedTotalSpend, "
    "NumActiveServices, AvgMonthlyChargePerService"
)

print("\nData preparation completed successfully.")

print("\n============================================")
print("FILES CREATED")
print("============================================")

print(
    "1. ATS_preprocessed_dataset.csv"
)

print(
    "2. ATS_X_train_scaled.csv"
)

print(
    "3. ATS_X_test_scaled.csv"
)

print(
    "4. ATS_y_train.csv"
)

print(
    "5. ATS_y_test.csv"
)

print(
    "6. standard_scaler.pkl"
)

print("\n============================================")
print("END OF DATA PREPARATION")
print("============================================")