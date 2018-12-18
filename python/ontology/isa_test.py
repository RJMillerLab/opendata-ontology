import json
import isa
import numpy as np
import random

labelTables = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_tables_20k.json', 'r'))
tableLables = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_20k.json', 'r'))
embSamples = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/samples/emb_20k.all', 'r'))
tableSamplesMap = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_20k.samples', 'r'))
labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/good_labels_20k.json', 'r'))
labelNames = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_20k.json', 'r'))
labelEmbs = {}

def random_samples_isa():
    random.seed(9001)
    l1embs = []
    inx1 = random.sample(range(0, len(embSamples)), 100)
    for i in inx1:
        fsi = [float(v) for v in embSamples[i]]
        l1embs.append(fsi)
    random.seed(3331)
    l2embs = []
    inx2 = random.sample(range(0, len(embSamples)), 75)
    for i in inx2:
        fsi = [float(v) for v in embSamples[i]]
        l2embs.append(fsi)
    print(isa.get_distance(np.asarray(l1embs), np.asarray(l2embs)))

def labels_pairwise_isa():
    for l in labels:
        labelEmbs[labelNames[str(l)]] = []
        for t in labelTables[str(l)]:
            if t not in tableSamplesMap:
                continue
            for si in tableSamplesMap[t]:
                fsi = [float(v) for v in embSamples[si]]
                labelEmbs[labelNames[str(l)]].append(fsi)
    print(len(labelEmbs))
    seen = {}
    for l1, es1 in labelEmbs.items():
        for  l2, es2 in labelEmbs.items():
            if l1!=l2 and l1+l2 not in seen:
                seen[l1+l2] = True
                seen[l2+l1] = True
                print("%s and %s" %(l1, l2))
                print(isa.get_distance(np.asarray(labelEmbs[l1]), np.asarray(labelEmbs[l2])))

random_samples_isa()
