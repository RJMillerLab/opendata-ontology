import csv
import json
import copy



tagSampleCFiles = dict()
domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/domains.json', 'r'))
TAG_SAMPLE_MAP = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/tag_samples_map.json'
SAMPLES_CSV_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/samples/v2/emb_samples.csv'
TAG_EMB_CSAMPLE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/tag_emb_csamples.json'
EMB_SAMPLE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/samples/emb.all'


csamples = []
dup = 0
# building all samples without labels
j = 0
tableLabels = dict()
tagSamples = dict()
for dom in domains:
    domainname = dom['name']
    j += 1
    if j % 50 == 0:
        print('processed %d domains.' % j)
    csample = ["0"] + list(dom['mean'])
    tag = dom['tag']
    # verifying if the sample has a duplicate
    i = 0
    try:
        i = csamples.index(csample)
        dup += 1
    except:
        csamples.append(csample)
        i = len(csamples)-1

    if tag not in tagSamples:
        tagSamples[tag] = []
    # verifying if the sample has a duplicate
    tagSamples[tag].append(i)
print('csamples: %d' % len(csamples))
print('%d duplicate samples' % dup)
json.dump(csamples, open(EMB_SAMPLE_FILE, 'w'))
# adding top labels to samples
tid = 0
for tag, sampleids in tagSamples.items():
    tid += 1
    if tid%20 == 0:
        print('saved samples of %d tags.' % tid)
    csf = open(SAMPLES_CSV_FILE.replace(".csv", "_" + tag + ".csv"), 'w')
    cswriter = csv.writer(csf, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)
    lcsamples = copy.deepcopy(csamples)
    for i in sampleids:
        lcsamples[i][0] = "1"
    cswriter.writerows(lcsamples)
    tagSampleCFiles[tag] = SAMPLES_CSV_FILE.replace(".csv", "_" + tag + ".csv")
    csf.close()
json.dump(tagSamples, open(TAG_SAMPLE_MAP, 'w'))
json.dump(tagSampleCFiles, open(TAG_EMB_CSAMPLE_FILE, 'w'))
