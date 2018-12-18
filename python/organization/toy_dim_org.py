#import random
import operator
import itertools
import numpy as np


def get_overlap(l1, l2):
    overlap = 0
    for t in label_tables[l1]:
        if t in label_tables[l2]:
            overlap += 1
    return overlap

#labels = ["2015", "2016", "2017", "2018", "ON", "QC", "MA", "CA", "NY", "BC"]
#tables = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
labels = ["2015", "2016", "2017", "2018", "NY", "BC","GA"]
tables = ["t1", "t2", "t3", "t4", "t5"]
table_labels = {}
label_tables = {}
table_labels = {"t1":["2015","BC"], "t2":["2016","NY"], "t3":["2017","NY"], "t4":["2016","BC"], "t5":["2017","BC"]}
label_tables = {"GA": [], "2018":[], "2015":["t1"], "2016":["t2","t4"], "2017":["t5","t3"], "NY":["t3"], "BC":["t1","t4","t5"]}
#for t in tables:
#    x = random.randint(1,len(labels))
#    table_labels[t] = list(np.asarray(labels)[random.sample(range(0, len(labels)), x)])
#    for l in table_labels[t]:
#        if l not in label_tables:
#            label_tables[l] = []
#        label_tables[l].append(t)
orgs = list(itertools.product([0, 1], repeat=len(labels)))
overlaps = {}
for il1 in range(len(labels)):
    overlaps[il1] = {}
    for il2 in range(len(labels)):
        if il1 == il2:
            continue
        o = get_overlap(labels[il1], labels[il2])
        overlaps[il1][il2] = o
        #if il2 not in overlaps:
        #    overlaps[il2] = {}
        #overlaps[il2][il1] = o
org_scores = {}
for io in range(len(orgs)):
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



