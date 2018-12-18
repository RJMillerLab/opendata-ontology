import json
import copy
import csv

tableLabels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/tables_31k.boosted_labels', 'r'))
labels = []
tables = {}
tcount = 0
for t, lps in tableLabels.items():
    if t not in tables:
        tables[t] = tcount
        tcount += 1
    for l, p in lps.items():
        if int(l) not in labels:
            labels.append(int(l))
print("num of labels: %d max of ids: %d" % (len(labels), max(labels)))
print("num of tables: %d" % len(tables))
csf = open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/samples/labels_samples.txt', 'w')
cswriter = csv.writer(csf, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)
samples = []
# the first element will filled with table id
sampleGen = [0] + [0.0 for l in labels]
for t, lps in tableLabels.items():
    table_id = tables[t]
    for l, p in lps.items():
        sample = copy.copy(sampleGen)
        sample[0] = table_id
        sample[int(l)+1] = p
        samples.append(sample)
cswriter.writerows(samples)
