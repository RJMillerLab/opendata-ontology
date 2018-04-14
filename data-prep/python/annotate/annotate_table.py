import numpy as np
import os
import json
from xgboost import XGBClassifier
from xgboost import Booster

def copy_table_labels(tls):
    tbls = {}
    for t, ls in tls.items():
        tbls[t] = {}
        for l in ls:
            tbls[t] = {l: 1.0}
    return tbls

LABEL_EMB_MODEL_FILE = os.environ['LABEL_EMB_MODEL_FILE']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
TABLE_BOOSTED_LABELS_FILE = os.environ['TABLE_BOOSTED_LABELS_FILE']
TABLE_SAMPLE_MAP = os.environ['TABLE_SAMPLE_MAP']
ALL_EMB_SAMPLE_FILE = os.environ['ALL_EMB_SAMPLE_FILE']

tableSamples = json.load(open(TABLE_SAMPLE_MAP, 'r'))
tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
modelFiles = json.load(open(LABEL_EMB_MODEL_FILE, 'r'))
samples = np.array(json.load(open(ALL_EMB_SAMPLE_FILE, 'r')))
tableBoostedLabels = copy_table_labels(tableLabels)

for l, mf in modelFiles.items():
    print('label: %s' % l)
    # load lable model
    model = XGBClassifier()
    booster = Booster()
    booster.load_model(mf)
    model._Booster = booster
    for t, ss in tableSamples.items():
        tss = samples[np.array(ss)]
        # predicting labels for tables
        preds = model.predict_proba(tss)
        tableBoostedLabels[t][int(l)] = float(max(list(preds[:,1])))
json.dump(tableBoostedLabels, open(TABLE_BOOSTED_LABELS_FILE, 'w'))

