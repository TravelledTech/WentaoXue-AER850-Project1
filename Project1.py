# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:13:15 2025

@author: wenta
"""

import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;
import seaborn as sns; 

# ================ 2.1: Data Precessing ================

data = pd.read_csv("Project 1 Data.csv");

# ================ 2.2: Data Visualization ================

#Offset the x axis slightly so they dont overlap 
plt.scatter(data['Step']-.05, data['X'], color='red', marker = 'x', s = 10, label='X')
plt.scatter(data['Step'], data['Y'], color='green', marker = 'd', s = 10, label='Y')
plt.scatter(data['Step']+.05, data['Z'], color='blue', marker = 'o', s = 10, label='Z')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 14))
plt.xlabel("Step")
plt.ylabel("Position")
plt.title("Training Data Position vs. Step")
plt.legend()
plt.show()

# ================ 2.3: Correlation Analysis ================

#Linear correlation
plt.figure()
corr_matrix = data.corr()
sns.heatmap(np.abs(corr_matrix))
plt.title("Pearson Correlation Matrix")

# ================ 2.4: Classification Model ================

# First Split Data
from sklearn.model_selection import StratifiedShuffleSplit

my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

x_data = data.drop('Step', axis = 1)
y_data = data['Step']

for train_index, test_index in my_splitter.split(x_data, y_data):
    x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

#Make the models ( (1) random forest, (2) Decision tree and (3) SVM) I dont think linear regression would work well
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# mdl1 = RandomForestClassifier(n_estimators=100, random_state=42)
# mdl2 = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=2, random_state=42)
# # Start with max depth of 5, I got no clue what min samples split and leaf should be, 
# mdl3 = SVC(kernel='rbf', C=1.0, gamma='scale') # Kernal = non-linear, c = ? and gamma = default?

# mdl1.fit(x_train, y_train)
# mdl2.fit(x_train, y_train)
# mdl3.fit(x_train, y_train)

# y_pred_train1 = mdl1.predict(x_train)
# y_pred_train2 = mdl2.predict(x_train)
# y_pred_train3 = mdl3.predict(x_train)

# from sklearn.metrics import mean_absolute_error

# mae_train1 = mean_absolute_error(y_pred_train1, y_train)
# mae_train2 = mean_absolute_error(y_pred_train2, y_train)
# mae_train3 = mean_absolute_error(y_pred_train3, y_train)

# print("Model 1 training MAE is: ", round(mae_train1,2)) #Overfit?
# print("Model 2 training MAE is: ", round(mae_train2,2))
# print("Model 3 training MAE is: ", round(mae_train3,2)) #Maybe Overfit?

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

# 1 and 2 will use gridsearch, 3 will use random

pipe1 = Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(random_state=42))])
pipe2 = Pipeline([("scaler", StandardScaler()),
                  ("clf", DecisionTreeClassifier(random_state=42))])
pipe3 = Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC())])

param_grid1 = {'clf__n_estimators': [5, 10, 20, 30, 50],    #Increase it slightly from the original
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10, 15],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__criterion': ['gini', 'entropy']}
param_grid2 = {'clf__max_depth': [None, 5, 10, 20], #Pretty much the same values as param1
    'clf__min_samples_split': [2, 5, 10, 15],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__criterion': ['gini', 'entropy']}
param_grid3 = {'clf__C': [0.1, 1, 10, 100, 1000],
    'clf__gamma': ['scale', 0.1, 0.01, 0.001],
    'clf__kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train grids
grid1 = GridSearchCV(
    pipe1,
    param_grid1,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
grid1.fit(x_train, y_train)

grid2 = GridSearchCV(
    pipe2,
    param_grid2,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
grid2.fit(x_train, y_train)

grid3 = RandomizedSearchCV(
    pipe3,
    param_grid3,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
grid3.fit(x_train, y_train)

# print("")
# print("Random Forest best params:", grid1.best_params_)
# print("Random Forest best CV score:", grid1.best_score_)
# print("")
# print("Decision Tree best params:", grid2.best_params_)
# print("Decision Tree best CV score:", grid2.best_score_)
# print("")
# print("SVM best params:", grid3.best_params_)
# print("SVM best CV score:", grid3.best_score_)
# print("")

# ================ 2.5: Model Performance ================

from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix

#Uses the best values from the previous part
y_pred1 = grid1.best_estimator_.predict(x_test)
y_pred2 = grid2.best_estimator_.predict(x_test)
y_pred3 = grid3.best_estimator_.predict(x_test)

f11 = f1_score(y_test, y_pred1, average='macro')
f12 = f1_score(y_test, y_pred2, average='macro')
f13 = f1_score(y_test, y_pred3, average='macro')

precision1 = precision_score(y_test, y_pred1, average='macro')
precision2 = precision_score(y_test, y_pred2, average='macro')
precision3 = precision_score(y_test, y_pred3, average='macro')

accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy3 = accuracy_score(y_test, y_pred3)

print("Random Forest F1: \t", f11, "\tprecision: ", precision1, "\taccuracy: ", accuracy1)
print("Decision Tree F1: \t", f12, "\tprecision: ", precision2, "\taccuracy: ", accuracy2)
print("SVM F1: \t\t\t", f13, "\tprecision: ", precision3, "\taccuracy: ", accuracy3)

#Create confusion matrix
cf1 = confusion_matrix(y_test, y_pred1)
cf2 = confusion_matrix(y_test, y_pred2)
cf3 = confusion_matrix(y_test, y_pred3)

#Plot each of the confusion matrix
plt.figure()
sns.heatmap(cf1)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.figure()
sns.heatmap(cf2)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.figure()
sns.heatmap(cf3)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# print("Random Forest Confusion Matrix")
# print(cf1)
# print("Decision Tree Confusion Matrix")
# print(cf2)
# print("SVM Confusion Matrix")
# print(cf3)

# ================ 2.6: Stacked Model Performance Analysis ================
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html

from sklearn.ensemble import StackingClassifier

estimator = [('rf', grid1.best_estimator_), ('dt', grid2.best_estimator_)]

from sklearn.linear_model import LogisticRegression

#I think I need to use logical regression

mdl4 = StackingClassifier(estimators = estimator, final_estimator = LogisticRegression())

mdl4.fit(x_train, y_train)

y_pred4 = mdl4.predict(x_test)

f14 = f1_score(y_test, y_pred4, average='macro')

precision4 = precision_score(y_test, y_pred4, average='macro')

accuracy4 = accuracy_score(y_test, y_pred4)

# print("Stacking Classifier F1: \t", f14, "\tprecision: ", precision4, "\taccuracy: ", accuracy4)

# ================ 2.7: Model Evaluation ================