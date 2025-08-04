# Data Cleaning and Preprocessing 

This project performs data cleaning and preprocessing on the Titanic dataset, as part of an AI & ML internship task. It focuses on cleaning raw data to make it suitable for machine learning models.

## 📂 Files in this Repository

- `data_cleaning_and_preprocessing.py` — Python script for data cleaning and preprocessing  

## 📊 Dataset Source

The Titanic dataset used in this task is publicly available:  
**[Titanic Dataset CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)**


## 🔍 What This Script Does

- Loads Titanic dataset directly from a URL
- Fills missing values:
  - `Age` with median
  - `Embarked` with mode
- Drops the `Cabin` column (too many missing values)
- Removes duplicate rows
- Encodes categorical variables:
  - Maps `Sex` to 0/1
  - One-hot encodes `Embarked`
- Standardizes numeric features: `Age`, `Fare` using `StandardScaler`
- Detects outliers in `Fare` and caps them using the IQR method
- Displays boxplot of the `Fare` column

---
## How to Run

Simple instructions on how to run the code.
Example:

Install necessary libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

## Run the script:
python datacleaningandpreprocessessing.py

## Technologies Used

-Python 3
-pandas numpy, matplotlib, seaborn, scikit-learn


