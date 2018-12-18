import networkx as nx
import csv

def relevance(d1, d2):
    intersect = len(set(d1).intersection(set(d2)))
    containment = float(intersect) / len(d1)
    jaccard = float(intersect) / (len(d1) + len(d2) - intersect)
    return (containment, jaccard)

# finding the lables that are potentially a part of the taxonomy of
source_labels = ['ckan_tags_society', 'ckan_tags_health', 'ckan_subject_government_and_politics', 'ckan_subject_nature_and_environment']
#source_labels = ['ckan_keywords_nature']
#source_labels = ['ckan_keywords_trends']
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
samples = {}
for t, lps in table_labels.items():
    sample = [0.0 for l in labels]
    include = False
    for l, p in lps.items():
        include = True
        sample[labels.index(l)] = p
        #sample[labels.index(l)] = 1.0
    if include:
        s = sum(sample)
        for i in range(len(sample)):
            if labels[i] not in samples:
                samples[labels[i]] = []
            samples[labels[i]].append(sample[i]/s)
print("num of samples: %d" % len(samples))
# creating label correlation graph
g = nx.Graph()
edges = []
# parent
for l1 in labels:
    # child
    for l2 in labels:
        if l1 != l2:
            rel = relevance(l1, l2)
            if rel[1] > 0.0:
                edges.append((l1, l2, rel[0]/rel[1]))
            rev_rel = relevance(l2, l1)
            if rev_rel[1] > 0.0:
                edges.append((l2, l1, rev_rel[0]/rev_rel[1]))
g.add_weighted_edges_from(edges)
print(len(g.nodes()))
print(len(g.edges()))
t=nx.maximum_spanning_tree(g)
print("num of edges: %d" % len(t.edges()))
print("num of nodes: %d" % len(t.nodes()))
for e in sorted(t.edges(data=True)):
    if e[2]['weight'] != 0.0:
        print(e)
#print(sorted(t.edges(data=True)))


