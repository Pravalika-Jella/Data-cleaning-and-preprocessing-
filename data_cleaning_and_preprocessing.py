

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Import the dataset and explore basic info
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("First 5 rows of the dataset:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nNumber of missing values (per column):")
print(df.isnull().sum())

# 2. Handle missing values using mean/median/imputation
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Optional: remove duplicates
df.drop_duplicates(inplace=True)

# 3. Convert categorical features into numerical using encoding
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. Normalize/standardize the numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 5. Visualize outliers using boxplots and remove them (using IQR capping for 'Fare')
plt.figure(figsize=(8,4))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_whisker = Q3 + 1.5 * IQR
lower_whisker = Q1 - 1.5 * IQR

df['Fare'] = np.where(df['Fare'] > upper_whisker, upper_whisker, df['Fare'])
df['Fare'] = np.where(df['Fare'] < lower_whisker, lower_whisker, df['Fare'])

print("\nFinal data info after cleaning:")
print(df.info())
print("\nSample cleaned data:")
print(df.head())
print("\nMissing values after cleaning (should be none):")
print(df.isnull().sum())
