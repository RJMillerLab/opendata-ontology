import numpy as np
import os
import json
from xgboost import XGBClassifier
from xgboost import Booster

def copy_table_labels(tls):
    tbls = {}
    for t, ls in tls.items():
        bls = {}
        for l in ls:
            bls[l] = 1.0
        tbls[t] = bls
    return tbls

LABEL_EMB_MODEL_FILE = os.environ['LABEL_EMB_MODEL_FILE']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
TABLE_BOOSTED_LABELS_FILE = os.environ['TABLE_BOOSTED_LABELS_FILE']
TABLE_SAMPLE_MAP = os.environ['TABLE_SAMPLE_MAP']
EMB_SAMPLE_FILE = os.environ['EMB_SAMPLE_FILE']
TEST_LABEL_NAMES_FILE = os.environ['TEST_LABEL_NAMES_FILE']
LABEL_BOOSTED_TABLES_FILE = os.environ['LABEL_BOOSTED_TABLES_FILE']
ATT_BOOSTED_LABELS_FILE = os.environ['ATT_BOOSTED_LABELS_FILE']
ATT_SAMPLE_MAP = os.environ['ATT_SAMPLE_MAP']


tableSamples = json.load(open(TABLE_SAMPLE_MAP, 'r'))
tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
modelFiles = json.load(open(LABEL_EMB_MODEL_FILE, 'r'))
samples = np.array(json.load(open(EMB_SAMPLE_FILE, 'r')))
tableBoostedLabels = copy_table_labels(tableLabels)
attBoostedLabels = dict()
label_names = json.load(open(TEST_LABEL_NAMES_FILE, 'r'))
attSamples = json.load(open(ATT_SAMPLE_MAP, 'r'))

for l, mf in modelFiles.items():
    print('label: %s' % l)
    # load lable model
    model = XGBClassifier()
    booster = Booster()
    booster.load_model(mf)
    model._Booster = booster
    for t, ss in tableSamples.items():
        #tss = samples[np.array(ss)]
        # predicting labels for tables
        #preds = model.predict_proba(tss)
        # predicting labels per att
        for si in ss:
            tss = [samples[si]]
            preds = model.predict_proba(tss)
            labelProb = np.float64(max(list(preds[:,1])))
            if labelProb != 0.0:
                print('new label: %s with %f' % (label_names[str(l)], labelProb))
                attname = t+'_'+str(attSamples[t][si])
                if attname not in attBoostedLabels:
                    attBoostedLabels[attname] = []
                if int(l) not in attBoostedLabels[attname]:
                #if int(l) not in tableBoostedLabels[t]:
                    #tableBoostedLabels[t][int(l)] = labelProb
                    attBoostedLabels[t+'_'+str(attSamples[t][si])].append(l)
#json.dump(tableBoostedLabels, open(TABLE_BOOSTED_LABELS_FILE, 'w'))
json.dump(attBoostedLabels, open(ATT_BOOSTED_LABELS_FILE, 'w'))
#tableBoostedLabels = json.load(open(TABLE_BOOSTED_LABELS_FILE, 'r'))
labelBoostedTables = {}
for t, lps in tableBoostedLabels.items():
    for l, p in lps.items():
        if l not in labelBoostedTables:
            labelBoostedTables[l] = []
        labelBoostedTables[l].append(t)
json.dump(labelBoostedTables, open(LABEL_BOOSTED_TABLES_FILE, 'w'))
