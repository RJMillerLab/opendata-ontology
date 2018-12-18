import networkx as nx
import pandas as pd
import csv
#from pgmpy.estimators import K2Score
from pgmpy.estimators import BicScore
from pgmpy.estimators import HillClimbSearch

# finding the lables that are potentially a part of the taxonomy of
#source_labels = ['ckan_tags_society', 'ckan_tags_health', 'ckan_subject_government_and_politics', 'ckan_subject_nature_and_environment']
source_labels = ['ckan_keywords_nature']
source_labels = ['ckan_keywords_trends']
#source_labels = [1023, 1418, 309, 45]
with open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_pairs.csv', 'r') as f:
    reader = csv.reader(f)
    edges = list(tuple([rec[1],rec[3]]) for rec in csv.reader(f, delimiter='|'))
    #edges = list(tuple([rec[0],rec[2]]) for rec in csv.reader(f, delimiter='|'))
g = nx.Graph()
g.add_edges_from(edges)
labels = []
for l in source_labels:
    if l not in labels:
        labels.append(l)
    for d in list(nx.descendants(g, l)):
        if d not in labels:
            labels.append(d)
print("num of reachable labels: %d" % len(labels))
# finding the tables that are associated with the taxonomy.
# preparing data for structure learning
with open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_table_names.csv', 'r') as f:
    reader = csv.reader(f)
    # skip the header
    next(reader, None)
    mems = list(tuple(rec) for rec in csv.reader(f, delimiter='|'))
# finding tables (samples)
# header: label_id|label_name|table_name|prob|source
table_labels = {}
for row in mems:
    if row[2] not in table_labels:
        table_labels[row[2]] = {}
    if row[1] in labels:
        table_labels[row[2]][row[1]] = float(row[3])
samples = []
for t, lps in table_labels.items():
    sample = [0.0 for l in labels]
    include = False
    for l, p in lps.items():
        include = True
        sample[labels.index(l)] = p
    if include:
        samples.append(sample)
print("num of samples: %d" % len(samples))
# learning strutcures
df = pd.DataFrame(samples, columns=labels)
df = df[df.columns]
print(len(df.columns))
#hc = HillClimbSearch(df, scoring_method=K2Score(df))
#hc = HillClimbSearch(df, scoring_method=BicScore(df))
#best_model = hc.estimate()
#for e in best_model.edges():
#    print(e)



