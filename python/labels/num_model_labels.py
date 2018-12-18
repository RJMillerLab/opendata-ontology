from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import os
import numpy as np
import json

SAMPLES_FILE = os.environ['NUM_SAMPLES_CSV_FILE']
PARAM_FILE = os.environ['NUM_PARAM_FILE']

# load data
data = read_csv(SAMPLES_FILE)
# shuffle data
data.sample(frac=1)
dataset = data.values
# split data into X and y
X = dataset[:,1:]
y = dataset[:,0]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
nclass = len(np.unique(label_encoded_y))
print(nclass)
# split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.20, random_state=42, stratify=label_encoded_y)
print(len(X_train))
print(len(y_test))
# parameters
params= json.load(open(PARAM_FILE, 'r'))
param_grid = {
    "estimator__learning_rate": params["etas"],
    "estimator__max_depth": params["max_depths"],
    "estimator__min_child_weight": params["min_child_weights"],
    "estimator__gamma": params["gammas"],
    "estimator__subsample": params["subsamples"],
    "estimator__reg_lambda": params["lambdas"],
    "estimator__reg_alpha": params["alphas"],
    "estimator__n_estimators": params["n_estimators"],
    "estimator__n_jobs": [20],
    "estimator__objective": ["multi:softmax"],
    "estimator__silent": [1],
}

model = OneVsRestClassifier(XGBClassifier(num_class=nclass, early_stopping_rounds=50, num_boost_round=1000))
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="f1_weighted", n_jobs=20, cv=kfold, verbose=10)
grid_result = grid_search.fit(X_train, y_train)
# summarize training results
print("Number of train samples: %d" % len(X_train))
print("Number of test samples: %d" % len(y_test))
print("Train Results:")
print("Best Validation of 5-fold (mean F1): %f using parameters %s" % (grid_result.best_score_, grid_result.best_params_))
# test results
test_pred = grid_search.predict(X_test)
test_pred_prob = grid_search.predict_proba(X_test)
print("Test Results:")
print("Accuracy: %.4g" % metrics.accuracy_score(y_test, test_pred))
print("Precision: ")
print(metrics.precision_score(y_test, test_pred, average=None))
print("Recall: ")
print(metrics.recall_score(y_test, test_pred, average=None))
print("F1: ")
print(metrics.f1_score(y_test, test_pred, average=None))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, test_pred, labels=[i for i in range(nclass)]))
