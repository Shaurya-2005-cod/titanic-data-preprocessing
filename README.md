# Titanic Data Preprocessing

## 📌 Objective
Clean and prepare the Titanic dataset by handling missing values, encoding categorical features, scaling numerical features, and removing outliers.

## 🛠 Tools Used
- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

## ✅ Steps Performed
1. Loaded dataset (`Titanic-Dataset.csv`)
2. Handled missing values:
   - Filled `Age` with median
   - Filled `Embarked` with mode
   - Dropped `Cabin`, `Name`, `Ticket`, `PassengerId`
3. Encoded `Sex` and `Embarked`
4. Scaled `Age` and `Fare` using `StandardScaler`
5. Removed outliers from `Fare` using IQR method
6. Saved cleaned data to `cleaned_titanic_dataset.csv`

## 📂 Files Included
- `Titanic-Dataset.csv` (original)
- `titanic_cleaning.py` (processing script)
- `cleaned_titanic_dataset.csv` (output)
- `README.md` (this file)

## 🔗 Submission
This project was completed as part of Task 1 for an AI & ML Internship.
