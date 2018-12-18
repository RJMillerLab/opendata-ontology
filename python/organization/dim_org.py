import operator
import json
import itertools
import numpy as np


def get_overlap(l1, l2):
    overlap = 0
    for t in label_tables[l1]:
        if t in label_tables[l2]:
            overlap += 1
    return overlap


good_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/good_labels_20k.json', 'r'))
label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_20k.json', 'r'))
all_label_tables = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_tables_20k.json', 'r'))
labels = []
label_tables = {}
table_labels = {}
for gl in good_labels:
    labels.append(label_names[str(gl)])
for l, ts in all_label_tables.items():
    if int(l) not in good_labels:
        continue
    label_tables[label_names[l]] = ts
    for t in ts:
        if t not in table_labels:
            table_labels[t] = []
        table_labels[t].append(l)

orgs = list(itertools.product([0, 1], repeat=len(labels)))
overlaps = {}
for il1 in range(len(labels)):
    overlaps[il1] = {}
    for il2 in range(il1+1,len(labels)):
        if il1 == il2:
            continue
        o = get_overlap(labels[il1], labels[il2])
        overlaps[il1][il2] = o
print('done computing label pair overlaps')
org_scores = {}
orgs = []
for io in range(len(orgs)):
    if io % 100 == 0:
        print('compted %d orgz' % io)
    o = list(orgs[io])
    nzs = np.nonzero(np.asarray(o))[0]
    if len(nzs) == 0:
        continue
    d1 = list(nzs)
    d2 = set(x for x in range(len(o))).difference(set(d1))
    org_scores[io] = 0
    for il1 in d1:
        for il2 in d2:
            org_scores[io] += overlaps[il1][il2]
sorted_oss = sorted(org_scores.items(), key=operator.itemgetter(1))
print(labels)
print(orgs[sorted_oss[-1:][0][0]])

