import csv
import networkx as nx
import sqlite3

TAXONOMY_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/mi_nature_tree_new.csv'
DB_NAME = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/org.db'
TABLE_NAME = 'mi_nature_tree'
LABEL_TABLES = 'label_table_names'
db = sqlite3.connect(DB_NAME)
cursor = db.cursor()
cursor.execute("drop table if exists nodes;")
cursor.execute("create table nodes as select parent as name from " + TABLE_NAME + " union select child as name from " + TABLE_NAME + ";")
cursor.execute("drop table if exists " + TABLE_NAME + "_leaves;")
cursor.execute("create table " + TABLE_NAME + "_leaves as select label_name, table_name from nodes,label_table_names where name=label_name;")
cursor.execute("drop table if exists " + TABLE_NAME + "_label_probs;")
cursor.execute("create table " + TABLE_NAME + "_label_probs as select label_name, 1.0*count(table_name)/(select count(*) from " + TABLE_NAME + "_leaves) as prob from " + TABLE_NAME + "_leaves group by label_name;")
roots = []
for label_name, prob in cursor.execute("select label_name, prob from " + TABLE_NAME + "_label_probs order by prob desc limit 10;").fetchall():
    roots.append(label_name)
table_labels = {}
label_tables = {}
for label_name, table_name in cursor.execute("select label_name, table_name from " +  LABEL_TABLES + ";").fetchall():
    if label_name not in label_tables:
        label_tables[label_name] = []
    label_tables[label_name].append(table_name)
    if table_name not in table_labels:
        table_labels[table_name] = []
    table_labels[table_name].append(label_name)
with open(TAXONOMY_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    edges = []
    for rec in csv.reader(f, delimiter='|'):
        edges.append((rec[0],rec[1]))
        if rec[0] in label_tables:
            label_tables[rec[0]].append(rec[1])
g = nx.Graph()
g.add_edges_from(edges)
print(len(g.edges()))
print(len(g.nodes()))
mdls = 0
for t, ls in table_labels.items():
    print('t: %s' % t)
    mdl = 0
    for l in ls:
        print('l: %s' % l)
        for r in roots:
            print('r: %s' % r)
            mdl += len(nx.shortest_path(g, source=r, target=l))
            mdl += len(label_tables[l])-1
    print("t: %d min desc: %d" % (t, mdl))
    mdls += mdl
print(mdls)
