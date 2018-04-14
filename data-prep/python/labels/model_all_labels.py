from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import json

PARAM_FILE = os.environ['EMB_PARAM_FILE']
LABEL_EMB_CSAMPLE_FILE = os.environ['LABEL_EMB_CSAMPLE_FILE']
MODEL_DIR = os.environ['MODEL_DIR']
LABEL_EMB_MODEL_FILE = os.environ['LABEL_EMB_MODEL_FILE']

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

labelModels = {}
sampleFiles = json.load(open(LABEL_EMB_CSAMPLE_FILE, 'r'))
for l, sf in sampleFiles.items():
    print("label %s" % l)
    # load data
    data = read_csv(sf)
    # shuffle data
    data.sample(frac=1)
    dataset = data.values
    # split data into X and y
    X = dataset[:,1:]
    y = dataset[:,0]
    # encode string class values as integers
    pos = list(y).count(1)
    print("Number of positive samples %d out of %d" % (pos, len(y)))
    if float(pos)/float(len(y)) < 0.02:
        print("Not enough samples for this lable.")
    nclass = len(np.unique(y))
    print(nclass)
    # split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(len(X_train))
    print(len(y_test))
    model = XGBClassifier(num_class=nclass, early_stopping_rounds=50, num_boost_round=1000)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=20, cv=kfold, verbose=10)
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
    print("Precision: %.4g" % metrics.average_precision_score(y_test, test_pred))
    print("Recall: %.4g" % metrics.recall_score(y_test, test_pred))
    print("F1: %.4g" % metrics.f1_score(y_test, test_pred))
    print("Confusion Matrix: [tn, fp, fn, tp]")
    print(metrics.confusion_matrix(y_test, test_pred).ravel())
    # saving the model
    grid_result.best_estimator_._Booster.save_model(os.path.join(MODEL_DIR, "label_" + l + ".model"))
    labelModels[l] = os.path.join(MODEL_DIR, "label_" + l + ".model")
    print("-----------------------")
json.dump(labelModels, open(LABEL_EMB_MODEL_FILE, 'w'))
