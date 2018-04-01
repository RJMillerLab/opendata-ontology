import os
import csv
import json

EMB_FILES_LIST = os.environ['EMB_FILES_LIST']
SAMPLES_FILE = os.environ['EMB_SAMPLES_FILE']
GOOD_LABELS_FILE = os.environ['GOOD_LABELS_FILE']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
OUTPUT_DIR = os.environ['OUTPUT_DIR']
TABLE_SAMPLE_MAP = os.environ['TABLE_SAMPLE_MAP']

sampleMap = {}
dinx = 0
tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
goodLabels = json.load(open(GOOD_LABELS_FILE, 'r'))
sf = open(SAMPLES_FILE, 'w')
swriter = csv.writer(sf, delimiter=' ', lineterminator='\n', quoting=csv.QUOTE_NONE)
for line in open(EMB_FILES_LIST):
    line = line.strip()
    tablename = os.path.dirname(line).replace("./", "")
    print(tablename)
    features = open(os.path.join(OUTPUT_DIR, "domains", line), 'r').readlines()[0].strip().split(',')
    sample = [""]
    for i in range(len(features)):
        sample.append(str(i) + ":" + features[i])
    for l in tableLabels[tablename]:
        if l in goodLabels:
            sample[0] = str(l)
            swriter.writerow(sample)
            if tablename not in sampleMap:
                sampleMap[tablename] = [dinx]
            else:
                sampleMap[tablename].append(dinx)
            print('label: ' + str(l))
            print(str(dinx))
            dinx += 1
    print('done processing ' + tablename)
sf.close()
json.dump(sampleMap, open(TABLE_SAMPLE_MAP, 'w'))
