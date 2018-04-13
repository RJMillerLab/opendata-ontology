import pandas as pd
import json
import numpy as np
from scipy.special import entr
import os

TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
GOOD_LABELS_FILE = os.environ['GOOD_LABELS_FILE']
LABELS_FILE = os.environ['LABELS_FILE']
K = int(os.environ['NUM_LABELS'])

label_names = json.load(open(LABELS_FILE, 'r'))
table_labels = json.load(open(TABLE_LABELS_FILE, 'r'))
labels = []
for t, ls in table_labels.items():
    labels.extend(ls)
df = pd.DataFrame(labels, columns = ['label'])
s = df['label'].value_counts()
labels = np.asarray(s.keys())
counts = np.asarray(s)
probs = counts/float(counts.sum())
entropy = entr(list(probs))
good_labels = labels[np.argsort(entropy)[-K:]].tolist()
good_labels.reverse()
print(good_labels)
good_probs = probs[np.argsort(entropy)[-K:]].tolist()
good_probs.reverse()
print(probs)
for l in good_labels:
    for k,v in label_names.items():
        if v == l:
            print(k)
json.dump(good_labels, open(GOOD_LABELS_FILE, 'w'))

