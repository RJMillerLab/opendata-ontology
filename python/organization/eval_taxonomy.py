import csv
import networkx as nx
import sqlite3
import math

TAXONOMY_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/mi_nature_tree_new.csv'
DB_NAME = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/org.db'
TABLE_NAME = 'mi_nature_tree'
db = sqlite3.connect(DB_NAME)
cursor = db.cursor()
cursor.execute("drop table if exists nodes;")
cursor.execute("create table nodes as select parent as name from " + TABLE_NAME + " union select child as name from " + TABLE_NAME + ";")
cursor.execute("drop table if exists " + TABLE_NAME + "_leaves;")
cursor.execute("create table " + TABLE_NAME + "_leaves as select label_name, table_name from nodes,label_table_names where name=label_name;")
cursor.execute("drop table if exists " + TABLE_NAME + "_label_probs;")
cursor.execute("create table " + TABLE_NAME + "_label_probs as select label_name, 1.0*count(table_name)/(select count(*) from " + TABLE_NAME + "_leaves) as prob from " + TABLE_NAME + "_leaves group by label_name;")
label_probs = {}
for label, prob in cursor.execute("select label_name, prob from " + TABLE_NAME + "_label_probs;").fetchall():
    label_probs[label] = float(prob)
roots = []
for label_name, prob in cursor.execute("select label_name, prob from " + TABLE_NAME + "_label_probs order by prob desc limit 10;").fetchall():
    roots.append(label_name)
print('label_probs')
print(len(label_probs))
with open(TAXONOMY_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    edges = []
    for rec in csv.reader(f, delimiter='|'):
        if rec[0] in roots and rec[1] in roots:
            continue
        if label_probs[rec[0]] > label_probs[rec[1]]:
            edges.append((rec[0],rec[1]))
        else:
            edges.append((rec[1],rec[0]))
g = nx.DiGraph()
g.add_edges_from(edges)
leaves = [x for x in g.nodes() if g.out_degree(x)==0]
print('leaves')
print(len(leaves))
print(leaves)
ancestors = {}
common_ancestors = {}
label_sims = {}
for n in list(leaves):
    ancestors[n] = nx.ancestors(g, n)
for n1 in list(leaves):
    for n2 in list(leaves):
        if n1 not in common_ancestors:
            common_ancestors[n1] = {}
        common_ancestors[n1][n2] = ancestors[n1].intersection(ancestors[n2])
print('common_ancestors')
print(len(common_ancestors))
cont_sum = 0
for in1 in range(len(leaves)):
    n1 = leaves[in1]
    for in2 in range(len(leaves)):
        n2 = leaves[in2]
        ics = []
        for l in common_ancestors[n1][n2]:
            ics.append(-1.0*math.log(label_probs[l]))
        if n1 not in label_sims:
            label_sims[n1] = {}
        if len(ics) != 0:
            #label_sims[n1][n2] = 10000
            #else:
            label_sims[n1][n2] = max(ics)
            cont_sum += label_sims[n1][n2]
print(cont_sum)
