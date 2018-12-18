import csv
import sqlite3
import graphviz as gv

FOREST_GRAPH_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/img/mi_nature_forest.dot'
FOREST_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/mi_nature_forest.csv'
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
taxonomy = gv.Graph(format='pdf')
csf = open(FOREST_FILE, 'w')
cswriter = csv.writer(csf, delimiter='|', lineterminator='\n', quoting=csv.QUOTE_NONE)
header = ['parent', 'child', 'mi', 'containment', 'jaccard', 'dotproduct', 'card1', 'card2', 'size1', 'size2']
cswriter.writerow(header)
with open(TAXONOMY_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    edges = []
    for rec in csv.reader(f, delimiter='|'):
        if rec[0] in roots and rec[1] in roots:
            continue
        cswriter.writerow(rec)
        taxonomy.edge(rec[0], rec[1])
filename = taxonomy.save(filename=FOREST_GRAPH_FILE)
print(filename)
