import networkx as nx
import math
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import json
import numpy as np


def discoverable_nodes(n):
    return list(nx.descendants(g, n))

def pick_roots(ns, root_num):
    descs = []
    #descs = list(active_nodes)
    #descs = list(g.nodes)
    for n in ns:
        ds = list(nx.ancestors(g, n))
        if len(ds) != 0:
            descs.extend(list(ds) + [n])
    inx = random.sample(range(0,len(descs)), min(len(descs),root_num))
    if len(inx) == 0:
        return []
    return list(np.array(descs)[np.array(inx)])

def pick_roots_plus(ns, root_num):
    descs = []
    roots = []
    for n in ns:
        ds = list(nx.descendants(g, n))
        roots.append(random.choice(ds))
        if len(ds) != 0:
            descs.extend(list(ds) + [n])
    inx = random.sample(range(0,len(descs)), min(len(descs),root_num))
    if len(inx) == 0:
        return []
    return list(np.array(descs)[np.array(inx)])



def evaluate_query(root, s):
    if root == s:
        return 1.0
    interprob = 1.0
    sp = 0.0
    count = 0
    if not nx.has_path(g,root,s):
        return math.pow(10, -15)#-1.0
    for p in nx.all_simple_paths(g, root, s, cutoff):
        path_prob = 1.0
        for i in range(len(p)-1):
            path_prob *= edges[p[i]][p[i+1]]
        interprob *= (1.0-path_prob)
        sp += path_prob
        count += 1
        if count == 50:
            return sp
    #prob = 1.0-interprob
    #return prob
    return sp


def evaluate(t, cs, isolated):
    print(t)
    if isolated:
        table_prob = (1.0/float(len(g.nodes())))
        print('unreachable tables: %f' % table_prob)
        return table_prob
    roots = pick_roots(cs, SIMULATION_NUM)
    print('num roots: %d' % len(roots))
    round_probs_intersect = 1.0
    table_prob = 0.0
    print('columns of table: %d' % len(cs))
    lps = []
    for root_column in roots:
        for query_column in cs:
            p = evaluate_query(root_column, query_column)
            #if p == -1.0:
            #    continue
            round_probs_intersect *= (1.0-p)
        table_prob += (1.0-round_probs_intersect)
        lps.append(1.0-round_probs_intersect)
    table_prob = table_prob/float(len(roots))
    print('table_prob: %f' % table_prob)
    return table_prob


def get_table_name(col_name):
    return col_name[:col_name.rfind('_')]

def init():
    sts = []
    for s, ts in edges.items():
        for t, p in ts.items():
            sts.append((s,t))

    g.add_edges_from(sts)

    # add all isolated nodes
    nodes = list(g.nodes())
    tcs = json.load(open(TABLE_ATTS, 'r'))
    for t, cs in tcs.items():
        for c in cs:
            if c not in nodes:
                g.add_node(c)
            else:
                active_nodes.append(c)

    oes = 0
    for c in tcs[tablename]:
        if c in edges:
            oes += len(edges[c])
    if oes == 0:
        return True
    else:
        return False

def init_plus_plus():
    tas = json.load(open(TABLE_ATTS))
    with open(TABLE_LIST, 'w') as f:
        for t in list(tas.keys()):
            print(t)
            f.write("%s\n" % t)


def init_plus():
    all_ts = json.load(open(TABLE_ATTS))
    all_es = json.load(open(SMALL_EDGE_FILE))
    inx = random.sample(range(0,len(all_ts)), 300)
    ts = list(np.array(list(all_ts.keys()))[np.array(inx)])
    small_ts = dict()
    for t, cs in all_ts.items():
        if t in ts:
            small_ts[t] = cs
    print('small ts: %d' % len(small_ts))
    json.dump(small_ts, open(SMALL_TABLE_ATTS, 'w'))
    small_edges = dict()
    count = 0
    for s, fs in all_es.items():
        if get_table_name(s) not in ts:
            continue
        small_edges[s] = dict()
        for f, p in fs.items():
            if get_table_name(f) not in ts:
                continue
            small_edges[s][f] = p
            count += 1
    print('number of edges: %d' % count)
    print('small_edges: %d' % len(small_edges))
    json.dump(small_edges, open(SMALL_EDGE_FILE, 'w'))


def plot():
    ps = list(json.load(open(NAVIGATION_PROBS, 'r')).values())
    tps = ps
    tps.sort()
    print(tps)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ps.sort()
    ax.set_ylim([0, 1.0])
    ax.plot([i for i in range(len(ps))], ps)
    plt.tight_layout()
    plt.savefig("ps_1000.pdf")
    plt.close()


TABLE_LIST = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/tables_1000.list'
TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts_1000'
SMALL_TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts_1000'
SMALL_EDGE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/edges_1000_t03'
#SMALL_EDGE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/edges_1000'
NAVIGATION_PROBS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/navigation_probs_1000'


cutoff = 10
g = nx.DiGraph()
edges = json.load(open(SMALL_EDGE_FILE, 'r'))
tablename = sys.argv[1]
active_nodes = []
isolated = init()
#init_plus()

seen_paths = dict()
round_seen_atts = dict()


ts = json.load(open(SMALL_TABLE_ATTS, 'r'))
print('number of tables: %d' % len(ts))
SIMULATION_NUM = 10

#RESULT_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/tableprobs_1000_t03.csv'
RESULT_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/tableprobs_1000_t03.csv'

tcs = json.load(open(TABLE_ATTS, 'r'))

print('tablename: %s' % tablename)

sf = open(RESULT_FILE, 'a', newline="\n")
delimiter = '|'
swriter = csv.writer(sf, delimiter=',', escapechar='\\', lineterminator='\n', quoting=csv.QUOTE_NONE)

tableprob = evaluate(tablename, tcs[tablename], isolated)

swriter.writerow([tablename, tableprob])

sf.close()

print('done')
