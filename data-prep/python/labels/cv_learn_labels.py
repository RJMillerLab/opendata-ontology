import numpy as np
import xgboost as xgb
import os
import json

SAMPLES_FILE = os.environ['EMB_SAMPLES_FILE']
LABEL_MODEL_SOFTMAX = os.environ['LABEL_MODEL_SOFTMAX']
LABEL_MODEL_SOFTPROB = os.environ['LABEL_MODEL_SOFTPROB']
PARAM_FILE = os.environ['PARAM_FILE']

# parameters
params= json.load(open(PARAM_FILE, 'r'))
etas = params["etas"]
max_depths = params["max_depths"]
num_boost_rounds = params["num_boost_rounds"]
min_child_weights = params["min_child_weights"]
early_stopping_rounds_vars = params["early_stopping_rounds_vars"]
gammas = params["gammas"]
subsamples = params["subsamples"]
lambdas = params["lambdas"]
alphas = params["alphas"]

nthread = 25
lam = 2
gridsearch_params = [
        (eta, max_depth, num_boost_round, min_child_weight, early_stopping_rounds, gamma, subsample, lam, alpha)
        for eta in etas
        for max_depth in max_depths
        for num_boost_round in num_boost_rounds
        for min_child_weight in min_child_weights
        for early_stopping_rounds in early_stopping_rounds_vars
        for gamma in gammas
        for subsample in subsamples
        for lam in lambdas
        for alpha in alphas
        ]

samples = xgb.DMatrix(SAMPLES_FILE)
labels = samples.get_label()
print('num samples: %d' % len(labels))
nclass = len(np.unique(labels))
for l in list(np.unique(labels)):
    print('%s: %d %f' %(l, list(labels).count(l),  float(list(labels).count(l))/float(len(labels))))

for eta, max_depth, num_boost_round, min_child_weight,                   early_stopping_rounds, gamma, subsample, lam, alpha in reversed(gridsearch_params):
    print("eta: %.2f, max_depth: %d, num_boost_round: %d, min_child_weight: %d, early_stopping_rounds: %d, gamma: %.2f, subsample: %d, lamda: %.2f, alpha: %.2f" % (eta, max_depth, num_boost_round, min_child_weight, early_stopping_rounds, gamma, subsample, lam, alpha))
    param = {'objective': 'multi:softmax', 'eta': eta, 'max_depth': max_depth, 'silent': 1, 'lambda': lam, 'nthread': nthread, 'num_class': nclass, 'min_child_weight': min_child_weight, 'gamma': gamma, 'subsample': subsample, 'lambda': lam, 'alpha': alpha}
#metrics={'mlogloss'}
#callbacks=[xgb.callback.print_evaluation(show_stdv=True)]
    cvresult = xgb.cv(param, samples, nfold=5, metrics={'merror'}, seed=0, num_boost_round=num_boost_round, stratified=True, early_stopping_rounds=early_stopping_rounds, callbacks=None)
    print(cvresult)
    print("----------------------------------")

