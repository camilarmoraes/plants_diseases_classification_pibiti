"""
kfold.py — Exemplo mínimo de K-Fold Cross Validation.

Demonstração didática do funcionamento do KFold do scikit-learn
com dados sintéticos, exibindo os índices de treino/teste para cada fold.
"""

import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([5, 12, 31, 74])
kf = KFold(n_splits=2)
#kf.get_n_splits(X)

#print(kf)

for train_index, test_index in kf.split(X,y):
    print("TRAIN: ",train_index, "TEST: ", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

