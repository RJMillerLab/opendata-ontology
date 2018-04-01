import numpy as np
import xgboost as xgb
import os

SAMPLES_FILE = os.environ['EMB_SAMPLES_FILE']
LABEL_MODEL_SOFTMAX = os.environ['LABEL_MODEL_SOFTMAX']
LABEL_MODEL_SOFTPROB = os.environ['LABEL_MODEL_SOFTPROB']

samples = xgb.DMatrix(SAMPLES_FILE)
nclass = len(np.unique(samples.get_label()))

param = {'objective': 'multi:softmax', 'eta': 0.1, 'max_depth': 100, 'silent': 1, 'nthread': 20, 'num_class': nclass}
num_round = 20

xgb.cv(param, samples, num_round, nfold=5, metrics={'merror'}, seed=0, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

