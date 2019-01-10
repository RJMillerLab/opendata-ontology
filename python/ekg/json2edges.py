import pandas as pd
import Levenshtein
import sqlite3
import emb
import copy
import json
import os
from datasketch.minhash import MinHash
from datasketch.lsh import MinHashLSH


def table2nodes(tablefullname):
    delimiter = '_'
    nodes_val = dict()
    tablefullname = tablefullname.replace('\n', '')
    try:
        #df = pd.read_json(os.path.join(OUTPUT, 'files', tablefullname), lines=True)
        df = pd.read_csv(os.path.join(OUTPUT, 'files', tablefullname))
        # dataframe preparation
        df.columns = [str(col).lower().replace('\s+', '_') for col in df.columns]
        df.replace("\s+", "_", regex=True, inplace=True)
        inx = 0
        for col in df.columns:
            tablename = os.path.dirname(tablefullname) + delimiter + os.path.basename(tablefullname)
            attname = os.path.dirname(tablefullname) + delimiter + os.path.basename(tablefullname) + delimiter + str(inx)
            df[col] = df[col].astype(str).str.lower().str.strip()
            es = set(df[col].tolist())
            nodes_val[attname] = es
            nodes_name[attname] = col
            if tablename not in table_atts:
                table_atts[tablename] = []
            table_atts[tablename].append(attname)
            inx += 1
    except Exception as e:
        print(e)
        print('failed to process ' + tablefullname)
        pass
    return nodes_val



def index_atts(atts):
    for name, fs in atts.items():
        m1 = MinHash(num_perm=128)
        for d in fs:
            m1.update(d.encode('utf8'))
        lsh.insert(name, m1)
        nodes_sig[name] = m1



def init():
    alldomains = json.load(open(DOMAIN_FILE))
    ts = dict()
    for dom in alldomains:
        if not dom['tag'].startswith('socrata_'):
            continue

        table = dom['name'][:dom['name'].rfind('_')]
        ts[table] = True
    print(len(ts))
    json.dump(list(ts.keys()), open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_tables.json', 'w'))


def process_tables():
    ts = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_tables.json', 'r'))
    print('num tables: %d' % len(ts))
    count = 0
    for t in ts:
        count += 1
        if count > 10:
            continue
        print('started processing %s' % t)
        ns = table2nodes(t)
        index_atts(ns)
    json.dump(table_atts, open(TABLE_ATTS, 'w'))


def get_pairs_semsyn_names():
    nameSimPairs = dict()
    atts = list(nodes_name.keys())
    for i in range(len(atts)):
        for j in range(i+1, len(atts)):
            sem = get_name_sim(nodes_name[atts[i]], nodes_name[atts[j]])
            if sem < 0.3:
                continue
            syn = Levenshtein.ratio(nodes_name[atts[i]], nodes_name[atts[j]])
            if syn < 0.2:
                continue
            if atts[i] not in nameSimPairs:
                nameSimPairs[atts[i]] = dict()
            if atts[j] not in nameSimPairs:
                nameSimPairs[atts[j]] = dict()
            nameSimPairs[atts[i]][atts[j]] = max(sem, syn)
            nameSimPairs[atts[j]][atts[i]] = max(sem, syn)
    return nameSimPairs



def get_name_sim(name1, name2):
    return emb.semantic_group_sim(db, name1, name2)


def get_pairs_jacc():

    simPairs = dict()
    seen = dict()
    for name, m in nodes_sig.items():
        result = lsh.query(m)
        for r in result:
            if r == name:
                continue
            if r+name not in seen and name+r not in seen:
                seen[r+name] = True
                seen[name+r] = True

                jacc = m.jaccard(nodes_sig[r])
                if jacc < 0.2:
                    continue

                if name not in simPairs:
                    simPairs[name] = dict()
                if r not in simPairs:
                    simPairs[r] = dict()
                simPairs[name][r] = jacc
                simPairs[r][name] = jacc
    json.dump(simPairs, open(JACC_SIM_FILE, 'w'))
    return simPairs


def get_pairs(synSimPairs, semSimPairs):
    combPairs = copy.deepcopy(semSimPairs)
    for n, js in synSimPairs.items():
        for m, j in js.items():
            combPairs[n][m] += j
    probPairs = copy.deepcopy(combPairs)
    for n, js in combPairs.items():
        for m, j in js.items():
            if combPairs[n][m] < 0.2:
                combPairs[n][m] = 0.0
            else:
                combPairs[n][m] /= 2.0
        ps = normalize(combPairs[n])
        for j, s in js.items():
            probPairs[n][j] = ps[j]

    json.dump(probPairs, open(JACC_EDGE_FILE, 'w'))


def normalize(ss):
    ps = dict()
    ssum = sum(list(ss.values()))
    for s, m in ss.items():
        ps[s] = m/ssum
    return ps


# Create LSH index
TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts'
DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs'
OUTPUT = '/home/fnargesian/FINDOPENDATA_DATASETS/10k'
JACC_EDGE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/jacc_edges'
JACC_SIM_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/jacc_sims'
DB_FILE = '/home/fnargesian/FASTTEXT/fasttext.db'
db = sqlite3.connect(DB_FILE)

nodes_sig = dict()
nodes_name = dict()
table_atts = dict()

lsh = MinHashLSH(threshold=0.2, num_perm=128)

#init()
process_tables()
get_pairs(get_pairs_jacc(), get_pairs_semsyn_names())

print('done')
