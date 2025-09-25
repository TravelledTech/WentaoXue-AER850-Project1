# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:13:15 2025

@author: wenta
"""

import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;
import seaborn as sns; 

# =========== 2.1: Data Precessing ===========

data = pd.read_csv("Project 1 Data.csv");

# =========== 2.2: Data Visualization ===========

#Offset the x axis slightly so they dont overlap 
plt.scatter(data['Step']-.05, data['X'], color='red', marker = 'x', s = 10, label='X')
plt.scatter(data['Step'], data['Y'], color='green', marker = 'd', s = 10, label='Y')
plt.scatter(data['Step']+.05, data['Z'], color='blue', marker = 'o', s = 10, label='Z')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 14))
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
plt.show

# =========== 2.3: Correlation Analysis ===========

#Linear correlation
plt.figure()
corr_matrix = data.corr()
sns.heatmap(np.abs(corr_matrix))
# X has high correlation even with that big jump in data?
# Shouldnt z be the most correlated?

# =========== 2.4: Classification Model ===========

#Might as well use k-fold (with stratified split)

# Start with stratified splitter
from sklearn.model_selection import StratifiedShuffleSplit

my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

x_data = data.drop('Step', axis = 1)
y_data = data['Step']

for train_index, test_index in my_splitter.split(x_data, y_data):
    x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold


