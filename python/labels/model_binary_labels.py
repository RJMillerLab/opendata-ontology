from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import json

SAMPLES_FILE = os.environ['EMB_SAMPLES_CSV_FILE']
LABEL_MODEL_SOFTMAX = os.environ['LABEL_MODEL_SOFTMAX']
LABEL_MODEL_SOFTPROB = os.environ['LABEL_MODEL_SOFTPROB']
PARAM_FILE = os.environ['PARAM_FILE']

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
    "learning_rate": params["etas"],
    "max_depth": params["max_depths"],
    "min_child_weight": params["min_child_weights"],
    "gamma": params["gammas"],
    "subsample": params["subsamples"],
    "reg_lambda": params["lambdas"],
    "reg_alpha": params["alphas"],
    "n_estimators": params["n_estimators"],
    "n_jobs": [20],
    "objective": ["multi:softmax"],
    "silent": [1],
}

model = XGBClassifier(num_class=nclass, early_stopping_rounds=50, num_boost_round=1000)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=20, cv=kfold, verbose=10)
grid_result = grid_search.fit(X_train, y_train)
# summarize training results
print("Number of train samples: %d" % len(X_train))
print("Number of test samples: %d" % len(y_test))
print("Train Results:")
print("Best Validation of 5-fold (mean F1): %f using parameters %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
# test results
test_pred = grid_search.predict(X_test)
test_pred_prob = grid_search.predict_proba(X_test)
print("Test Results:")
print("Accuracy: %.4g" % metrics.accuracy_score(y_test, test_pred))
print("Precision: %.4g" % metrics.average_precision_score(y_test, test_pred))
print("Recall: %.4g" % metrics.recall_score(y_test, test_pred))
print("F1: %.4g" % metrics.f1_score(y_test, test_pred))
print("Confusion Matrix: [tn, fp, fn, tp]")
print(metrics.confusion_matrix(y_test, test_pred).ravel())
