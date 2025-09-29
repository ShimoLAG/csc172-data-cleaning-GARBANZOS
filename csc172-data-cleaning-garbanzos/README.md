# Data Cleaning with AI Support

## Student Information
- Name: LOUISE ANTONDY GARBANZOS
- Course Year: BSCS 4
- Date: 2025-08-29

## Dataset
- Source: [\[Kaggle/UCI link\]](https://www.kaggle.com/competitions/titanic/data)
- Name: test.csv

## Issues found
- Missing values:
    Age: 177 missing

    Cabin: 687 missing

    Embarked: 2 missing

- Duplicates: 0 found
- Inconsistencies:
    Column names not standardized

    String values (Sex, Embarked) needed consistent formatting

    Outliers: Fare had extreme values (max = 512.33), skewing distribution

## Cleaning steps
1. Missing values:

    Age filled with median (robust to outliers)

    Embarked filled with mode (“S”, most common)

    Cabin dropped (too many missing values, ~77%)

2. Duplicates: None so no action required
3. Inconsistencies:
    Converted column names to lowercase
    Cleaned string values for consistency
4. Outliers:
    Used IQR method on Fare

    Capped extreme values at upper bound (reduced max from 512 → 65.63)

## AI prompts used
- Prompt 1: ""Can you give me the code for cleaning the Titanic training dataset, and explain to me each line of code and what they're used for, including handling missing values, duplicates, standardizing formats, detecting/treating outliers, and showing before/after snapshots?""
- Generated code: (paste)
 import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("train.csv")

# --- Exploratory checks (before cleaning) ---
print("Shape (before):", df.shape)         # shows rows × columns before cleaning
print("\nSample rows (before):")           
print(df.head())                           # preview first 5 rows
print("\nInfo (before):")
print(df.info())                           # column data types + non-null counts
print("\nSummary statistics (before):")
print(df.describe(include='all'))          # numeric + categorical summary
print("\nMissing values (before):")
print(df.isna().sum())                     # count missing per column
print("\nNumber of duplicates (before):", df.duplicated().sum())

# --- Cleaning steps ---
# 1. Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())     # numeric → fill with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # categorical → mode
df = df.drop(columns=['Cabin'])                      # drop Cabin (too many missing)

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Standardize formats
df.columns = df.columns.str.lower()                  # column names → lowercase
df['sex'] = df['sex'].str.strip().str.lower()        # clean Sex values
df['embarked'] = df['embarked'].str.strip().str.upper()  # clean Embarked codes

# 4. Outlier treatment (Fare)
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
df['fare'] = np.where(df['fare'] > upper, upper, df['fare'])  # cap outliers

# --- Exploratory checks (after cleaning) ---
print("\nShape (after):", df.shape)
print("\nSample rows (after):")
print(df.head())
print("\nSummary statistics (after):")
print(df.describe(include='all'))
print("\nMissing values (after):")
print(df.isna().sum())


## Results
- Rows before: 891
- Columns before: 12
- Rows after: 891
- Columns after: 11 (Cabin dropped)
- Missing values after cleaning: 0
- Fare (mean): reduced from 32.20 → 24.04
- Fare (max): reduced from 512.33 → 65.63

Video: link