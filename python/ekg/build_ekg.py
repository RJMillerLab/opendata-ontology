import pandas as pd
import random
import numpy as np
import Levenshtein
import sqlite3
import emb
import copy
import json
import os
from datasketch.minhash import MinHash
from datasketch.lsh import MinHashLSH


def table2nodes_attnames(tablefullname):
    delimiter = '_'
    nodes_val = dict()
    tablefullname = tablefullname.replace('\n', '')
    try:
        df = pd.read_csv(os.path.join(OUTPUT, 'files', tablefullname))
        # dataframe preparation
        df.columns = [str(col).lower().replace('\s+', '_') for col in df.columns]
        df.replace("\s+", "_", regex=True, inplace=True)
        inx = 0
        for col in df.columns:
            tablename = os.path.dirname(tablefullname) + delimiter + os.path.basename(tablefullname)
            attname = os.path.dirname(tablefullname) + delimiter + os.path.basename(tablefullname) + delimiter + str(inx)
            #df[col] = df[col].astype(str).str.lower().str.strip()
            #es = set(df[col].tolist())
            #nodes_val[attname] = es
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

def table2nodes_attvalues(tablefullname):
    delimiter = '_'
    nodes_val = dict()
    tablefullname = tablefullname.replace('\n', '')
    try:
        df = pd.read_csv(os.path.join(OUTPUT, 'files', tablefullname))
        # dataframe preparation
        df.columns = [str(col).lower().replace('\s+', '_') for col in df.columns]
        df.replace("\s+", "_", regex=True, inplace=True)
        inx = 0
        for col in df.columns:
            #tablename = os.path.dirname(tablefullname) + delimiter + os.path.basename(tablefullname)
            attname = os.path.dirname(tablefullname) + delimiter + os.path.basename(tablefullname) + delimiter + str(inx)
            #df[col] = df[col].astype(str).str.lower().str.strip()
            #es = set(df[col].tolist())
            es = df[col].tolist()
            nodes_val[attname] = es
            #nodes_name[attname] = col
            #if tablename not in table_atts:
            #    table_atts[tablename] = []
            #table_atts[tablename].append(attname)
            inx += 1
    except Exception as e:
        print(e)
        print('failed to process ' + tablefullname)
        pass
    return nodes_val



def index_atts(atts):
    for name, fs in atts.items():
        m1 = MinHash(num_perm=128)
        count = 0
        for d in fs:
            count += 1
            if count > 200:
                continue
            m1.update(str(d).encode('utf8'))
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
    inx = random.sample(range(0,len(ts)), 500)
    ekg_tables = list(np.array(list(ts.keys()))[np.array(inx)])
    json.dump(ekg_tables, open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg_socrata_tables.json', 'w'))


def process_tables():
    ts = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg_socrata_tables.json', 'r'))
    print('num tables: %d' % len(ts))
    count = 0
    for t in ts:
        count += 1
        if count%100==0:
            print('processed %d tables' % count)
        table2nodes_attnames(t)
        #ns = table2nodes_attvalues(t)
        #index_atts(ns)
        #print('done processing %s' % t)
    print('saving graph')
    json.dump(table_atts, open(TABLE_ATTS, 'w'))
    print('done saving')


def get_table_name(col_name):
    return col_name[:col_name.rfind('_')]


def get_pairs_semsyn_names():
    print('get_pairs_semsyn_names')
    nameSimPairs = dict()
    atts = list(nodes_name.keys())
    for i in range(len(atts)):
        if (i+1) % 200 == 0:
            print('processed %d names' % i)
        name_vec1 = emb.semantic(db, nodes_name[atts[i]])
        table_name1 = get_table_name(atts[i])

        for j in range(i+1, len(atts)):
            table_name2 = get_table_name(atts[j])
            if table_name1 == table_name2:
                continue
            syn = Levenshtein.ratio(nodes_name[atts[i]], nodes_name[atts[j]])
            if syn < 0.2:
                continue
            name_vec2 = emb.semantic(db, nodes_name[atts[j]])
            sem = get_name_sim(name_vec1, name_vec2)
            if sem < 0.3:
                continue

            if atts[i] not in nameSimPairs:
                nameSimPairs[atts[i]] = dict()
            if atts[j] not in nameSimPairs:
                nameSimPairs[atts[j]] = dict()
            nameSimPairs[atts[i]][atts[j]] = max(sem, syn)
            nameSimPairs[atts[j]][atts[i]] = max(sem, syn)
    json.dump(nameSimPairs, open(NAME_SIM_FILE, 'w'))
    return nameSimPairs



def get_name_sim(name1, name2):
    return emb.semantic_group_sim(name1, name2)


def get_pairs_jacc():
    print('get_pairs_jacc')
    simPairs = dict()
    seen = dict()
    print('nodes_sig: %d' % len(nodes_sig))
    count = 0
    for name, m in nodes_sig.items():
        count += 1
        if count%200 == 0:
            print('processed %d sigs' % count)
        result = lsh.query(m)
        table1 = get_table_name(name)
        for r in result:
            if r == name:
                continue
            if get_table_name(r) == table1:
                continue
            if r+name not in seen:
                seen[r+name] = True
                seen[name+r] = True

                jacc = m.jaccard(nodes_sig[r])
                #if jacc < 0.6:
                if jacc < 0.3:
                    continue

                if name not in simPairs:
                    simPairs[name] = dict()
                if r not in simPairs:
                    simPairs[r] = dict()
                simPairs[name][r] = jacc
                simPairs[r][name] = jacc
    json.dump(simPairs, open(JACC_SIM_FILE, 'w'))
    return simPairs


def get_pairs_small(synSimPairsFile, semSimPairsFile):
    small_tables = json.load(open(SMALL_TABLE_ATTS))
    synSimPairs = json.load(open(synSimPairsFile))
    semSimPairs = json.load(open(semSimPairsFile))
    print('get_pairs')
    #combPairs = copy.deepcopy(semSimPairs)
    combPairs = dict()
    for n, ss in semSimPairs.items():
        if get_table_name(n) not in small_tables:
            continue
        combPairs[n] = dict()
        for m, s in ss.items():
            if get_table_name(m) not in small_tables:
                continue
            combPairs[n][m] = s
    for n, js in synSimPairs.items():
        if get_table_name(n) not in small_tables:
            continue
        if n not in combPairs:
            combPairs[n] = dict()
        for m, j in js.items():
            if get_table_name(m) not in small_tables:
                continue
            if m in combPairs[n]:
                combPairs[n][m] += j
                #combPairs[n][m] = max(combPairs[n][m], j)
            else:
                #print('syn: %f sem < 0.3' % j)
                combPairs[n][m] = j
    probPairs = copy.deepcopy(combPairs)
    for n, js in combPairs.items():
        for m, j in js.items():
            #if combPairs[n][m] < 1.8:
            if combPairs[n][m] < 0.7:
                del(probPairs[n][m])
                if len(probPairs[n]) == 0:
                    del(probPairs[n])
            else:
                probPairs[n][m] /= 2.0
        if n not in probPairs:
            continue
        ps = normalize(probPairs[n])
        for j, s in probPairs[n].items():
            probPairs[n][j] = ps[j]
    ss = 0.0
    # number of outgoing edges on average
    oes = [len(ns) for m, ns in probPairs.items()]
    print('avg outgoing edges: %f' % (sum(oes)/len(probPairs)))
    print('min: %d max: %d' % (min(oes),max(oes)))
    print('num nodes: %d' % len(probPairs))
    print('node_names: %d' % len(nodes_name))

    json.dump(probPairs, open(EDGE_FILE_SMALL, 'w'))




def get_pairs(synSimPairsFile, semSimPairsFile):
    synSimPairs = json.load(open(synSimPairsFile))
    semSimPairs = json.load(open(semSimPairsFile))
    print('get_pairs')
    combPairs = copy.deepcopy(semSimPairs)
    for n, js in synSimPairs.items():
        if n not in combPairs:
            combPairs[n] = dict()
        for m, j in js.items():
            if m in combPairs[n]:
                combPairs[n][m] += j
            else:
                print('syn: %f sem < 0.3' % j)
                combPairs[n][m] = j
    probPairs = copy.deepcopy(combPairs)
    for n, js in combPairs.items():
        for m, j in js.items():
            #if combPairs[n][m] < 0.9:
            if combPairs[n][m] < 0.6:
                del(probPairs[n][m])
                if len(probPairs[n]) == 0:
                    del(probPairs[n])
            else:
                probPairs[n][m] /= 2.0
        if n not in probPairs:
            continue
        ps = normalize(probPairs[n])
        for j, s in probPairs[n].items():
            probPairs[n][j] = ps[j]
    ss = 0.0
    # number of outgoing edges on average
    for m, ns in probPairs.items():
        ss += len(ns)
    print('avg outgoing edges: %f' % (ss/len(probPairs)))
    print('num nodes: %d' % len(probPairs))
    print('node_names: %d' % len(nodes_name))

    json.dump(probPairs, open(EDGE_FILE, 'w'))


def normalize(ss):
    ps = dict()
    ssum = sum(list(ss.values()))
    for s, m in ss.items():
        ps[s] = m/ssum
    return ps


# Create LSH index
TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts_500'
DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs'
OUTPUT = '/home/fnargesian/FINDOPENDATA_DATASETS/10k'
EDGE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/edges_500'
#EDGE_FILE_SMALL = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/edges_300'
EDGE_FILE_SMALL = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/edges_1000_t03'
JACC_SIM_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/jacc_sims_1000'
NAME_SIM_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/name_sims_1000'
DB_FILE = '/home/fnargesian/FASTTEXT/fasttext.db'
db = sqlite3.connect(DB_FILE)
SMALL_TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts_1000'
#SMALL_TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/tables_831.json'


nodes_sig = dict()
nodes_name = dict()
table_atts = dict()

lsh = MinHashLSH(threshold=0.2, num_perm=128)

#init()
#process_tables()

#get_pairs_semsyn_names()
#get_pairs_jacc()

get_pairs_small(JACC_SIM_FILE, NAME_SIM_FILE)
#get_pairs(JACC_SIM_FILE, NAME_SIM_FILE)

print('done')
