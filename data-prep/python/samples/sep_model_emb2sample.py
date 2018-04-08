import os
import csv
import json

EMB_FILES_LIST = os.environ['EMB_FILES_LIST']
SAMPLES_CSV_FILE = os.environ['EMB_SAMPLES_CSV_FILE']
SAMPLES_DMAT_FILE = os.environ['EMB_SAMPLES_DMAT_FILE']
GOOD_LABELS_FILE = os.environ['GOOD_LABELS_FILE']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
OUTPUT_DIR = os.environ['OUTPUT_DIR']
TABLE_SAMPLE_MAP = os.environ['TABLE_SAMPLE_MAP']
LABEL_EMB_CSAMPLE_FILE = os.environ['LABEL_EMB_CSAMPLE_FILE']
LABEL_EMB_DSAMPLE_FILE = os.environ['LABEL_EMB_DSAMPLE_FILE']

labelSampleCFiles = {}
labelSampleDFiles = {}
labelTables = {}
sampleMap = {}
dinx = 0
tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
goodLabels = json.load(open(GOOD_LABELS_FILE, 'r'))
# building a map from labels to tables
for t, ls in tableLabels.items():
    for l in ls:
        if l in labelTables:
            labelTables[l].append(t)
        else:
            labelTables[l] = [t]
tableSamples = {}
csamples = []
dsamples = []
count = 0
# building all samples without labels
for line in open(EMB_FILES_LIST):
    line = line.strip()
    tablename = os.path.dirname(line).replace("./", "")
    if tablename not in tableLabels:
        continue
    features = open(os.path.join(OUTPUT_DIR, "domains", line), 'r').readlines()[0].strip().split(',')
    if len(features) == 0:
        continue
    csample = ["0"]
    dsample = ["0"]
    for i in range(len(features)):
        dsample.append(str(i) + ":" + features[i])
        csample.append(features[i])
    # verifying if the sample has a duplicate
    if csample not in csamples:
        dsamples.append(dsample)
        csamples.append(csample)
    else:
        count += 1
    if tablename in tableSamples:
        tableSamples[tablename].append(len(csamples)-1)
    else:
        tableSamples[tablename] = [len(csamples)-1]
print('%d duplicate samples' % count)
for l in goodLabels:
    csf = open(SAMPLES_CSV_FILE.replace(".csv", "_" + str(l) + ".csv"), 'w')
    dsf = open(SAMPLES_DMAT_FILE.replace(".dmat", "_" + str(l) + ".dmat"), 'w')
    cswriter = csv.writer(csf, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)
    dswriter = csv.writer(dsf, delimiter=' ', lineterminator='\n', quoting=csv.QUOTE_NONE)
    lcsamples = csamples[:]
    ldsamples = dsamples[:]
    for t in labelTables[l]:
        if t not in tableSamples:
            continue
        for i in tableSamples[t]:
            lcsamples[i][0] = "1"
            ldsamples[i][0] = "1"
    cswriter.writerows(lcsamples)
    dswriter.writerows(ldsamples)
    labelSampleCFiles[l] = SAMPLES_CSV_FILE.replace(".csv", "_" + str(l) + ".csv")
    labelSampleDFiles[l] = SAMPLES_DMAT_FILE.replace(".dmat", "_" + str(l) + ".dmat")
    csf.close()
    dsf.close()
json.dump(tableSamples, open(TABLE_SAMPLE_MAP, 'w'))
json.dump(labelSampleCFiles, open(LABEL_EMB_CSAMPLE_FILE, 'w'))
json.dump(labelSampleDFiles, open(LABEL_EMB_DSAMPLE_FILE, 'w'))
