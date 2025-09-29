import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("Data\\train.csv")


# BEFORE CLEANING SNAPSHOTS

print("Shape (before):", df.shape)
print("\nSample rows (before):")
print(df.head())

print("\nInfo (before):")
df.info()

print("\nSummary statistics (before):")
print(df.describe(include='all'))

print("\nMissing values (before):")
print(df.isnull().sum())

print("\nNumber of duplicates (before):", df.duplicated().sum())


# 2. CLEANING STEPS


# a) Handle missing values
# 'Age' → fill with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# 'Embarked' → fill with mode (most common port)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 'Cabin' → too many missing, drop column
df = df.drop(columns=['Cabin'])

# b) Remove duplicates
df = df.drop_duplicates()

# c) Standardize formats
# Lowercase column names
df.columns = df.columns.str.lower().str.strip()

# Make 'sex' and 'embarked' consistent
df['sex'] = df['sex'].str.lower().str.strip()
df['embarked'] = df['embarked'].str.upper().str.strip()  # C, Q, S

# d) Detect & treat outliers in 'Fare' using IQR
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Option: Cap extreme fares instead of dropping
df['fare'] = df['fare'].clip(lower, upper)


# AFTER CLEANING SNAPSHOTS

print("\nShape (after):", df.shape)
print("\nSample rows (after):")
print(df.head())

print("\nSummary statistics (after):")
print(df.describe(include='all'))

print("\nMissing values (after):")
print(df.isnull().sum())


# 3. SAVE CLEANED DATASET

df.to_csv("Data/cleaned_dataset.csv", index=False)
#print("\n✅ Cleaned dataset saved as Data/cleaned_dataset.csv")
