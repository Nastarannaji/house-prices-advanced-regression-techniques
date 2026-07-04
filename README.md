# house-prices-advanced-regression-techniques
End-to-end machine learning project for predicting house prices using Python, Pandas, Scikit-learn, and Linear Regression.
# House Price Prediction using Machine Learning

## Project Overview

This project predicts residential house prices using machine learning and the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The goal is to build an end-to-end machine learning pipeline that cleans real-world data, preprocesses numerical and categorical features, trains a regression model, evaluates its performance, and generates predictions for unseen houses.

This project was built to practice the complete machine learning workflow rather than simply training a model.

---

## Dataset

Dataset: **House Prices - Advanced Regression Techniques**

The dataset contains detailed information about residential homes, including:

* Lot size
* Neighborhood
* House style
* Number of rooms
* Garage information
* Basement information
* Exterior quality
* Year built
* Overall quality
* And many other property features

Target variable:

* **SalePrice** – the selling price of the house

The original dataset can be found on Kaggle:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## Project Workflow

### 1. Data Exploration (EDA)

* Loaded the training dataset
* Explored the dataset using:

  * `head()`
  * `info()`
  * `describe()`
  * `shape()`
* Identified missing values
* Examined feature types

---

### 2. Data Cleaning

* Removed columns with excessive missing values
* Filled missing numerical values using the median
* Filled missing categorical values with `"None"`

---

### 3. Feature Engineering

* Converted categorical features into numerical values using:

  * Label Encoding
  * One-Hot Encoding (`pd.get_dummies()`)

* Prepared the feature matrix for machine learning

---

### 4. Model Training

Split the dataset into:

* Training set
* Validation set

Used **Linear Regression** from Scikit-learn to train the model.

---

### 5. Model Evaluation

Performance metrics:

* Mean Absolute Error (MAE)
* R² Score

Current model performance:

* **MAE:** ~20,600
* **R² Score:** ~0.88

These results indicate that the model explains approximately 88% of the variance in house prices.

---

### 6. Prediction

The trained model is used to predict house prices for the unseen Kaggle test dataset.

Predictions are exported into a submission file compatible with the Kaggle competition.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## Machine Learning Concepts Practiced

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Missing Value Handling
* Label Encoding
* One-Hot Encoding
* Feature Engineering
* Train/Test Split
* Linear Regression
* Model Evaluation
* Prediction on Unseen Data

---

## Future Improvements

Potential improvements include:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost
* LightGBM
* CatBoost
* Hyperparameter tuning
* Cross-validation
* Feature selection
* Feature importance visualization

---

## Project Structure

```text
house-price-prediction/
│
├── train.csv
├── test.csv
├── sample_submission.csv
├── end-to-end.py
├── submission.csv
└── README.md
```

---

## Results

This project demonstrates the complete machine learning pipeline from raw data to prediction.

Although the initial model uses Linear Regression, the project establishes a strong baseline that can be extended with more advanced regression algorithms for improved performance.

---

## Author

Created as a machine learning portfolio project to practice building complete end-to-end regression models using real-world housing data.
