import os
import csv
import json

NUM_FILES_LIST = os.environ['NUM_FILES_LIST']
SAMPLES_CSV_FILE = os.environ['NUM_SAMPLES_CSV_FILE']
SAMPLES_DMAT_FILE = os.environ['NUM_SAMPLES_DMAT_FILE']
GOOD_LABELS_FILE = os.environ['GOOD_LABELS_FILE']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
OUTPUT_DIR = os.environ['OUTPUT_DIR']
TABLE_SAMPLE_MAP = os.environ['TABLE_SAMPLE_MAP']

sampleMap = {}
dinx = 0
tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
goodLabels = json.load(open(GOOD_LABELS_FILE, 'r'))
csf = open(SAMPLES_CSV_FILE, 'w')
dsf = open(SAMPLES_DMAT_FILE, 'w')
cswriter = csv.writer(csf, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)
dswriter = csv.writer(dsf, delimiter=' ', lineterminator='\n', quoting=csv.QUOTE_NONE)
for line in open(NUM_FILES_LIST):
    line = line.strip()
    tablename = os.path.dirname(line).replace("./", "")
    if tablename not in tableLabels:
        continue
    features = open(os.path.join(OUTPUT_DIR, "domains", line), 'r').readlines()[0].strip().split(',')
    if len(features) == 0:
        continue
    csample = [""]
    dsample = [""]
    for i in range(len(features)):
        dsample.append(str(i) + ":" + features[i])
        csample.append(features[i])
    for l in tableLabels[tablename]:
        if l in goodLabels:
            # class number is the index of the corresponding label.
            # this is because xgboost accepts 0 to number of classes class labels.
            if l == 290:
                csample[0] = "1"
                dsample[0] = "1"
            elif 290 not in tableLabels[tablename]:
                csample[0] = "0"
                dsample[0] = "0"
            #csample[0] = str(goodLabels.index(l))
            #dsample[0] = str(goodLabels.index(l))
            if dsample[0] != "":
                cswriter.writerow(csample)
                dswriter.writerow(dsample)
                if tablename not in sampleMap:
                    sampleMap[tablename] = [dinx]
                else:
                    sampleMap[tablename].append(dinx)
                dinx += 1
csf.close()
dsf.close()
json.dump(sampleMap, open(TABLE_SAMPLE_MAP, 'w'))
