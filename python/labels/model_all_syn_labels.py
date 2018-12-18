from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import json

PARAM_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/labels/params_10k_100.json'
LABEL_EMB_CSAMPLE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/tag_emb_csamples.json'
MODEL_DIR = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/models/v2'
LABEL_EMB_MODEL_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/labels_365.models'
TEST_RESULTS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/model_results_365.json'

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

test_reuslts = {}
labelModels = {}
sampleFiles = json.load(open(LABEL_EMB_CSAMPLE_FILE, 'r'))
print("number of labels: %d" % len(sampleFiles))
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
    neg_samples_inx = [i for i, x in enumerate(list(y)) if x == 0]
    pos_samples_inx = [i for i, x in enumerate(list(y)) if x == 1]
    if len(pos_samples_inx) == 1:
        print('only one sample: skip %s!' % l)
        continue
    neg = list(y).count(0)
    pos = list(y).count(1)
    neg_sample_size = min(neg, max(20*pos, 1000))
    Xp, yp = [], []
    for i in pos_samples_inx:
        Xp.append(X[i])
        yp.append(y[i])
    for j in range(len(neg_samples_inx)):
        if j >= neg_sample_size:
            continue
        i = neg_samples_inx[j]
        Xp.append(X[i])
        yp.append(y[i])

    X = np.array(Xp)
    y = np.array(yp)
    print("# pos samples was %d and now is %d." % (pos, list(y).count(1)))
    print("# neg samples was %d and now is %d." % (neg, list(y).count(0)))
    print("Number of positive samples %d out of %d" % (pos, len(y)))
    if float(pos)/float(len(y)) < 0.02:
        print("Not enough samples for this lable.")
        #continue
    # get a subset of positive samples when too many
    nclass = len(np.unique(y))
    # split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print('#train: %d #test: %d' % (len(X_train), len(y_test)))
    model = XGBClassifier(num_class=nclass, early_stopping_rounds=50, num_boost_round=1000)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=20, cv=kfold, verbose=10)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize training results
    print("Train Results:")
    print("Best Validation of 5-fold (mean F1): %f using parameters %s" % (grid_result.best_score_, grid_result.best_params_))
    # test results
    test_pred = grid_search.predict(X_test)
    test_pred_prob = grid_search.predict_proba(X_test)
    for p in list(test_pred_prob[:,1]):
        if p!=0.0 and p!=1.0:
            print(p)
    acc = metrics.accuracy_score(y_test, test_pred)
    prec = metrics.average_precision_score(y_test, test_pred)
    recall = metrics.recall_score(y_test, test_pred)
    f1 = metrics.f1_score(y_test, test_pred)
    print("Test Results:")
    print("Accuracy: %.4g" % acc)
    print("Precision: %.4g" % prec)
    print("Recall: %.4g" % recall)
    print("F1: %.4g" % f1)
    print("Confusion Matrix: [tn, fp, fn, tp]")
    print(metrics.confusion_matrix(y_test, test_pred).ravel())
    test_reuslts[l] = {'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1}
    # saving the model
    if metrics.recall_score(y_test, test_pred) < 0.4:
        print('the recall is too low for this classifier.')
    grid_result.best_estimator_._Booster.save_model(os.path.join(MODEL_DIR, "label_" + l + ".model"))
    labelModels[l] = os.path.join(MODEL_DIR, "label_" + l + ".model")
    print("-----------------------")
json.dump(labelModels, open(LABEL_EMB_MODEL_FILE, 'w'))
print(len(labelModels))
json.dump(test_reuslts, open(TEST_RESULTS_FILE, 'w'))
