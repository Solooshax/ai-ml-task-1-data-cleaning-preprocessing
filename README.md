# AI & ML Internship – Task 1
## Data Cleaning & Preprocessing

## Overview

This project completes **Task 1: Data Cleaning & Preprocessing**.


## Objectives

- Explore a raw dataset
- Inspect null values and data types
- Handle missing values
- Convert categorical features into numerical form
- Detect and handle outliers
- Standardize numerical features
- Validate the final ML-ready dataset

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
ai-ml-task-1-data-cleaning-preprocessing/
├── data/
│   └── employee_attrition_raw.csv
├── notebooks/
│   └── Task_1_Data_Cleaning_Preprocessing.ipynb
├── src/
│   └── preprocessing.py
├── outputs/
│   ├── 01_missing_values.png
│   ├── 02_outliers_before.png
│   ├── 03_outliers_after.png
│   └── 04_correlation_heatmap.png
├── README.md
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
└── .gitignore
```

## Preprocessing Workflow

1. Load the dataset
2. Inspect shape, data types and statistics
3. Analyze missing values
4. Fill numerical missing values using the median
5. Fill categorical missing values using the mode
6. Remove the employee identifier
7. Encode categorical variables using one-hot encoding
8. Visualize numerical outliers using boxplots
9. Detect outliers using the IQR method
10. Remove detected outliers
11. Standardize numerical features with `StandardScaler`
12. Validate the final dataset
13. Visualize correlations

## Why this dataset?

The internship task permits any relevant dataset. This dataset was designed specifically for the exercise so the repository remains self-contained and demonstrates missing values, categorical variables, numerical features and outliers without requiring an external download.



## Conclusion

The project demonstrates a complete data-cleaning and preprocessing workflow from raw data to a numerical, standardized, ML-ready dataset.
