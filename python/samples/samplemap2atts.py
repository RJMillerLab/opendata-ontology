import json
import ntpath
import os.path

embfiles = open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/emb_31k.files', 'r').read().splitlines()
tablesamples = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_31k.samples', 'r'))
sampletotableatt = dict()

i = 0
ne = 1
for ef in embfiles:
    features = open(os.path.join('/home/fnargesian/FINDOPENDATA_DATASETS/10k/domains', ef)).readlines()[0]. strip().split(',')
    if len(features) == 0:
        continue
    tablename = os.path.dirname(ef).replace("./", "")
    if tablename not in tablesamples:
        ne += 1
        continue
    if tablename not in sampletotableatt:
        i = 0
        sampletotableatt[tablename] = dict()
    print(len(tablesamples[tablename]))
    if i >= len(tablesamples[tablename]):
        print('continue')
        continue
    sampletotableatt[tablename][int(tablesamples[tablename][i])] = int(ntpath.basename(ef).replace('.ft-mean', ''))
    i+=1
json.dump(sampletotableatt, open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/samples_to_tableatt_31k.map', 'w'))
print(ne)
