import numpy as np
import json
import pickle
import xgboost as xgb
import os
from random import shuffle

SAMPLES_FILE = os.environ['EMB_SAMPLES_FILE']
LABEL_MODEL_SOFTMAX = os.environ['LABEL_MODEL_SOFTMAX']
LABEL_MODEL_SOFTPROB = os.environ['LABEL_MODEL_SOFTPROB']
PARAM_FILE = os.environ['PARAM_FILE']

# parameters
params= json.load(open(PARAM_FILE, 'r'))
etas = params["etas"]
max_depths = params["max_depths"]
num_rounds = params["num_rounds"]
num_boost_rounds = params["num_boost_rounds"]
min_child_weights = params["min_child_weights"]
early_stopping_rounds_vars = params["early_stopping_rounds_vars"]
n_estimators = 1000
nthread = 25

samples = xgb.DMatrix(SAMPLES_FILE)
nclass = len(np.unique(samples.get_label()))
sample_inx = [i for i in range(samples.num_row())]

for eta in etas:
    for max_depth in max_depths:
        for num_round in num_rounds:
            for num_boost_round in num_boost_rounds:
                for min_child_weight in min_child_weights:
                    for early_stopping_rounds in early_stopping_rounds_vars:
                        print("eta: %f, max_depth: %d, num_round: %d, early_stopping_rounds: %d, num_boost_round: %d, min_child_weight: %d, early_stopping_rounds: %d" % (eta, max_depth, num_round, early_stopping_rounds, num_boost_round, min_child_weight, early_stopping_rounds))
                        shuffle(sample_inx)
                        train_inx = sample_inx[:int(0.7 * samples.num_row())]
                        test_inx = sample_inx[int(0.7 * samples.num_row()):]
                        xg_train = samples.slice(train_inx)
                        xg_test = samples.slice(test_inx)
                        print('num of train: ' + str(xg_train.num_row()))
                        print('num of test: ' + str(xg_test.num_row()))
                        # setup parameters for xgboost
                        param = {'objective': 'multi:softmax', 'eta': eta, 'max_depth': max_depth, 'silent': 1, 'nthread': 20, 'num_class': nclass, 'min_child_weight': min_child_weight}

                        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
                        print('training with softmax')
                        bst = xgb.train(param, xg_train, num_round, watchlist)
                        # save model
                        pickle.dump(bst, open(LABEL_MODEL_SOFTMAX, "wb"))
                        # get prediction
                        print('predicting with softmax')
                        pred = bst.predict(xg_test)
                        error_rate = np.sum(pred != xg_test.get_label()) / xg_test.get_label().shape[0]
                        print('Test error using softmax = {}'.format(error_rate))

                        # do the same thing again, but output probabilities
                        print('training with probs')
                        param['objective'] = 'multi:softprob'
                        bst = xgb.train(param, xg_train, num_round, watchlist)
                        # save model
                        pickle.dump(bst, open(LABEL_MODEL_SOFTPROB, "wb"))
                        print('predicting with probs')
                        pred_prob = bst.predict(xg_test).reshape(xg_test.get_label().shape[0], nclass)
                        pred_label = np.argmax(pred_prob, axis=1)
                        error_rate = np.sum(pred_label != xg_test.get_label()) / xg_test.get_label().shape[0]
                        print("eta: %f, max_depth: %d, num_round: %d" % (eta, max_depth, num_round))
                        print('Test error using softprob = {}'.format(error_rate))
                        print('------------------------------------')
