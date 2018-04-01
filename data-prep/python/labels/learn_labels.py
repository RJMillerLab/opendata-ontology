import numpy as np
import pickle
import xgboost as xgb
import os
#from random import shuffle

SAMPLES_FILE = os.environ['EMB_SAMPLES_FILE']
LABEL_MODEL_SOFTMAX = os.environ['LABEL_MODEL_SOFTMAX']
LABEL_MODEL_SOFTPROB = os.environ['LABEL_MODEL_SOFTPROB']

samples = xgb.DMatrix(SAMPLES_FILE)
print('loaded')
nclass = len(samples.get_label())
print('nclass')
print(nclass)
# dividing data into test and train
sample_inx = [i for i in range(samples.num_row())]
print('sample_inx')
#shuffle(sample_inx)
train_inx = sample_inx[:int(0.7 * samples.num_row())]
test_inx = sample_inx[int(0.7 * samples.num_row()):]
print('here1')
xg_train = samples.slice(train_inx)
xg_test = samples.slice(test_inx)
print('num of train: ' + str(xg_train.num_row()))
print('num of test: ' + str(xg_test.num_row()))
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 10
param['num_class'] = nclass

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 10
print('training with softmax')
bst = xgb.train(param, xg_train, num_round, watchlist)
# save model
pickle.dump(bst, open(LABEL_MODEL_SOFTMAX, "wb"))
# get prediction
print('predicting with softmax')
pred = bst.predict(xg_test)
error_rate = np.sum(pred != xg_train.get_label()) / xg_train.get_label().shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
print('training with probs')
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# save model
pickle.dump(bst, open(LABEL_MODEL_SOFTPROB, "wb"))
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
print('predicting with probs')
pred_prob = bst.predict(xg_test).reshape(xg_test.get_label().shape[0], nclass)
pred_label = np.argmax(pred_prob, axis=1)
print(pred_label)
print(xg_test.get_label())
error_rate = np.sum(pred_label != xg_test.get_label()) / xg_test.get_label().shape[0]
print('Test error using softprob = {}'.format(error_rate))
