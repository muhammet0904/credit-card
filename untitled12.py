# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:07:52 2025

@author: mami
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("C:/Users/mami/datacampcsv/projects/credit_card/cc_approvals.data", header=None)

print(df.info())

for col in df.columns:
    print(f"{col} sütununda boş değer sayısı: ",df[col].isna().sum())

df = df.replace("?", np.NaN)

for col in df.columns:
    print(f"{col} sütununda boş değer sayısı: ",df[col].isna().sum())


df_copy = df.copy()


cols_to_float = [1,2,7,10,12] 


for col in cols_to_float:
    df[col] = df[col].astype("float")

print(df.info())



for col in df.columns:
    if df[col].dtypes == "object":
        df[col] = df[col].fillna(df[col].value_counts().index[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

for col in df.columns:
    print(f"{col} sütununda boş değer sayısı: ",df[col].isna().sum())

df_dummies = pd.get_dummies(df, drop_first= True)

X = df_dummies.iloc[:, :-1].values
y = df_dummies.iloc[:, [-1]].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rescaled_X_train = scaler.fit_transform(X_train)
rescaled_X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(rescaled_X_train, y_train)
y_train_pred = logreg.predict(rescaled_X_train)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, y_train_pred))



from sklearn.model_selection import GridSearchCV

tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol= tol, max_iter= max_iter)

grid_model = GridSearchCV(logreg, param_grid= param_grid, cv= 5)
grid_model_result = grid_model.fit(rescaled_X_train, y_train)
 

best_train_score, best_train_params = grid_model_result.best_score_, grid_model_result.best_params_
print(f"best: {best_train_score} using {best_train_params}")

best_model = grid_model_result.best_estimator_
best_score = best_model.score(rescaled_X_test, y_test)
print("Accuracy of logistic regression classifier: ", best_score)















