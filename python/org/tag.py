import pandas as pd
import json
import numpy as np
from scipy.special import entr
import os
import glob

TABLE_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'
GOOD_LABELS_FILE = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/output/good_labels.json'
LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'
OPENDATA_DIR = '/home/fnargesian/FINDOPENDATA_DATASETS/10k'
EMBS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_embs'
DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs'
LABEL_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_label_embs'
K = 1000
FT_DIM = 300


def get_good_labels():
    print('get_good_labels')
    label_names = json.load(open(LABELS_FILE, 'r'))
    print('label_names: %d' % len(label_names))
    table_labels = json.load(open(TABLE_LABELS_FILE, 'r'))
    print('tables: %d' % len(table_labels))
    labels = []
    for t, ls in table_labels.items():
        embCount = len(glob.glob1(os.path.join(OPENDATA_DIR, 'domains', t),"*.ft-mean"))
        for i in range(embCount):
            labels.extend(ls)

    df = pd.DataFrame(labels, columns = ['label'])
    s = df['label'].value_counts()
    labels = np.asarray(s.keys())
    counts = np.asarray(s)
    probs = counts/float(counts.sum())
    entropy = entr(list(probs))
    good_labels = labels[np.argsort(entropy)[-K:]].tolist()
    good_labels.reverse()
    good_labels = [int(l) for l in good_labels]
    good_probs = probs[np.argsort(entropy)[-K:]].tolist()
    good_probs.reverse()
    good_label_names = []
    for l in good_labels:
        good_label_names.append(label_names[str(l)])
    json.dump(good_label_names, open(GOOD_LABELS_FILE, 'w'))


def tag_embs():
    label_names = json.load(open(LABELS_FILE, 'r'))
    table_labels = json.load(open(TABLE_LABELS_FILE, 'r'))
    label_embs = dict()
    label_emb_counts = dict()
    domains = []
    df = pd.read_csv(EMBS_FILE)
    count = 0
    print('total number of tables: %d' % len(table_labels))
    print('number of rows: %d' % len(df))
    domcount = 0
    tablenotfound = []

    for index, row in df.iterrows():
        count += 1
        if (count+1) % 50 == 0:
            print('processed %d rows ().' % count)
        #if count > 20000:
        #    continue
        e = []
        t = row['dataset_name'].replace('\\\"', '')
        for i in range(FT_DIM):
            e.append(float(row['f' + str(i)]))
        # summing emb vectors
        if t not in table_labels:
            #print('%s is not in table_labels.' % t)
            if t not in tablenotfound:
                tablenotfound.append(t)
            continue
        for l in table_labels[t]:
            l = str(l)
            if not l.startswith('socrata_'):
                continue
            domcount += 1
            continue
            # creating a domain
            dom = dict()
            c = str(row['column_id']).replace('\\\"', '')
            dom['name'] = t + '_' + c
            dom['tag'] = label_names[l]
            dom['mean'] = list(e)
            domains.append(dom)

            if label_names[l] in label_embs:
                label_embs[l] = label_embs[l] + np.array(e)
                label_emb_counts[l] += 1
            else:
                label_embs[l] = np.array(e)
                label_emb_counts[l] = 1

    label_embs2 = dict()
    for l, e in label_embs.items():
        label_embs2[label_names[l]] = list(label_embs[l] / label_emb_counts[l])

    print('%d tables out of %d do not have lables.' % (len(tablenotfound), len(table_labels)))
    json.dump(domains, open(DOMAIN_FILE, 'w'))
    json.dump(label_embs2, open(LABEL_EMB_FILE, 'w'))
    print('done %d  %d' % (len(domains), len(label_embs2)))


def filter_tags():
    label_names = json.load(open(LABELS_FILE))
    print('label_names: %d' % len(label_names))
    socrata_labels = [int(i) for i, l in label_names.items() if l.startswith('socrata_')]
    print('some_labels: %d' % len(socrata_labels))
    table_labels = json.load(open(TABLE_LABELS_FILE))
    socrata_tables = []
    for t, ls in table_labels.items():
        for l in ls:
            if l in socrata_labels and t not in socrata_tables:
                socrata_tables.append(t)
    print('tables: %d' % len(table_labels))
    print('socrata tables: %d' % len(socrata_tables))



#filter_tags()

tag_embs()



