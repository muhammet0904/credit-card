# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:07:52 2025

@author: mami
"""

# Credit Card Approval Prediction Project
# This project aims to clean and prepare credit card application data and build a logistic regression model.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset (no header in this file)
df = pd.read_csv("C:/Users/mami/datacampcsv/projects/credit_card/cc_approvals.data", header=None)

# Step 2: Inspect the dataset
print(df.info())

# Step 3: Check missing values before processing
for col in df.columns:
    print(f"{col} sütununda boş değer sayısı: ",df[col].isna().sum())

# Step 4: Replace "?" with NaN for easier handling of missing values
df = df.replace("?", np.NaN)

# Step 5: Check missing values again
for col in df.columns:
    print(f"{col} sütununda boş değer sayısı: ",df[col].isna().sum())

# Backup copy before modifications
df_copy = df.copy()

# Step 6: Convert selected columns to float type for numerical processing
cols_to_float = [1,2,7,10,12] 
for col in cols_to_float:
    df[col] = df[col].astype("float")

# Check updated data types
print(df.info())

# Step 7: Fill missing values
# For categorical (object) columns: fill with most frequent value
# For numerical columns: fill with column mean
for col in df.columns:
    if df[col].dtypes == "object":
        df[col] = df[col].fillna(df[col].value_counts().index[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Final check for any remaining missing values
for col in df.columns:
    print(f"{col} sütununda boş değer sayısı: ",df[col].isna().sum())

# Step 8: Encode categorical variables using one-hot encoding
df_dummies = pd.get_dummies(df, drop_first=True)

# Step 9: Define features and target
X = df_dummies.iloc[:, :-1].values
y = df_dummies.iloc[:, [-1]].values 

# Step 10: Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 11: Standardize features for better model performance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rescaled_X_train = scaler.fit_transform(X_train)
rescaled_X_test = scaler.transform(X_test)

# Step 12: Build and train a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(rescaled_X_train, y_train)
y_train_pred = logreg.predict(rescaled_X_train)

# Step 13: Evaluate the model with confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, y_train_pred))

# Step 14: Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)

grid_model = GridSearchCV(logreg, param_grid=param_grid, cv=5)
grid_model_result = grid_model.fit(rescaled_X_train, y_train)

# Step 15: Show best parameters and accuracy score on test data
best_train_score, best_train_params = grid_model_result.best_score_, grid_model_result.best_params_
print(f"Best training score: {best_train_score} using parameters: {best_train_params}")

best_model = grid_model_result.best_estimator_
best_score = best_model.score(rescaled_X_test, y_test)
print("Accuracy of logistic regression classifier on test data: ", best_score)














