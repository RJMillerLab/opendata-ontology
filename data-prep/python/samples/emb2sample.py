import os
import csv
import json

EMB_FILES_LIST = os.environ['EMB_FILES_LIST']
SAMPLES_FILE = os.environ['EMB_SAMPLES_FILE']
GOOD_LABELS_FILE = os.environ['GOOD_LABELS_FILE']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
OUTPUT_DIR = os.environ['OUTPUT_DIR']
TABLE_SAMPLE_MAP = os.environ['TABLE_SAMPLE_MAP']
#NUM_TABLE_SAMPLES = os.environ['NUM_TABLE_SAMPLES']

tableGoodLabels = {}
sampleMap = {}
dinx = 0
tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
goodLabels = json.load(open(GOOD_LABELS_FILE, 'r'))
for gl in goodLabels:
    tableGoodLabels[gl] = []
sf = open(SAMPLES_FILE, 'w')
swriter = csv.writer(sf, delimiter=' ', lineterminator='\n', quoting=csv.QUOTE_NONE)
for line in open(EMB_FILES_LIST):
    #if len(sampleMap) > int(NUM_TABLE_SAMPLES):
    #    continue
    line = line.strip()
    tablename = os.path.dirname(line).replace("./", "")
    tableGoodLabels[tablename] = []
    if tablename not in tableLabels:
        continue
    features = open(os.path.join(OUTPUT_DIR, "domains", line), 'r').readlines()[0].strip().split(',')
    if len(features) == 0:
        continue
    sample = [""]
    print(tablename)
    for i in range(len(features)):
        print(features[i])
        sample.append(str(i) + ":" + features[i])
    for l in tableLabels[tablename]:
        if l in goodLabels:
            # class number is the index of the corresponding label.
            # this is because xgboost accepts 0 to number of classes class labels.
            if l == 290:
                sample[0] = str(1)
            else:
                sample[0] = str(0)
            #sample[0] = str(goodLabels.index(l))
            swriter.writerow(sample)
            if tablename not in sampleMap:
                sampleMap[tablename] = [dinx]
            else:
                sampleMap[tablename].append(dinx)
            dinx += 1
sf.close()
json.dump(sampleMap, open(TABLE_SAMPLE_MAP, 'w'))
for t, c in tableGoodLabels.items():
    if len(c) > 1:
        print(t)
