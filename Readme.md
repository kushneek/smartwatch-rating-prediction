# Smartwatch Rating Prediction 
# Machine Learning Regression Project

## Project Overview

This project aims to predict smartwatch customer ratings using Machine Learning based on various product specifications such as brand, price, battery life, display size, strap material, touchscreen support, Bluetooth availability, and other product attributes.

The primary objective of this project is to understand and implement the complete end-to-end Machine Learning workflow, including data cleaning, exploratory data analysis, feature engineering, preprocessing, model training, and model evaluation.

Currently, the project implements **Linear Regression** as the baseline regression model. Additional regression algorithms will be implemented and compared in future updates.

---

## Dataset

**Source:** Kaggle

The dataset contains smartwatch product information including:

- Brand
- Current Price
- Original Price
- Discount Percentage
- Number of Ratings
- Dial Shape
- Strap Color
- Strap Material
- Touchscreen
- Battery Life
- Bluetooth
- Display Size
- Weight
- Customer Rating (Target Variable)

---

## Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
One-Hot Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
```

---

## Project Structure

```
smartwatch-rating-prediction/

│
├── Dataset/
│   ├── smartwatches.csv
│   ├── cleaned_dataset.csv
│   └── Processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── comparison.csv
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   └── 05_Model_Evaluation.ipynb
│
├── models/
│   └── Linear_Regression.joblib
│   └── Decision_Tree_Model.joblib
│   └── Ridge_Model.joblib
│
├── src/
│
├── Requirements.txt
│
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## Data Preprocessing

The following preprocessing techniques were performed:

- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Train-Test Split
- One-Hot Encoding
- Feature Scaling
- Model Serialization using Joblib

---

## Model Implemented

- Linear Regression
- Decision Tree
- Ridge Model

---

## Model Performance

| Metric | Score |
|---------|-------:|
| Mean Absolute Error (MAE) | 0.2818 |
| Mean Squared Error (MSE) | 0.1669 |
| Root Mean Squared Error (RMSE) | 0.4085 |
| R² Score | 0.3966 |
| Adjusted R² Score | 0.2245 |

---

## Key Findings

- The model predicts smartwatch ratings with an average error of approximately **0.28 rating points**.
- The Linear Regression model explains approximately **39.66%** of the variance in smartwatch ratings.
- After accounting for model complexity, the Adjusted R² decreases to **22.45%**, indicating that several features contribute limited predictive information.
- Residual analysis suggests that customer ratings cannot be fully explained using a simple linear relationship.
- Linear Regression serves as a strong baseline model for comparison with more advanced regression algorithms.

---

## Future Improvements

The following regression algorithms will be implemented and compared:

- Decision Tree Regression
- Random Forest Regression
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors (KNN) Regression
- Gradient Boosting Regression
- XGBoost Regression

Future enhancements will also include:

- Hyperparameter Tuning
- Cross Validation
- Model Comparison
- Feature Importance Analysis
- Model Deployment using Flask

---

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Linear Regression
- Regression Evaluation Metrics
- Model Persistence using Joblib
- End-to-End Machine Learning Pipeline

---

## Author

**Kushagra Neekhra**

B.Tech Computer Science Engineering

Machine Learning | Data Science | Artificial Intelligence
