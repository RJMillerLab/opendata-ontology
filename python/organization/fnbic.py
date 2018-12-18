import networkx as nx
import csv
import math
import numpy as np
import graphviz as gv

def entropy(counts):
    ps = counts/float(np.sum(counts))  # coerce to float and normalize
    ps = ps[np.nonzero(ps)]            # toss out zeros
    H = -sum(ps * np.log2(ps))   # compute entropy
    return H

def mutual_info(x, y, bins):
    counts_xy = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])[0]
    counts_x  = np.histogram(x, bins=bins, range=[0, 1])[0]
    counts_y  = np.histogram(y, bins=bins, range=[0, 1])[0]
    H_x  = entropy(counts_x)
    H_y  = entropy(counts_y)
    H_xy = entropy(counts_xy)
    return H_x + H_y - H_xy

# finding the lables that are potentially a part of the taxonomy of
#source_labels = ['ckan_tags_society', 'ckan_tags_health', 'ckan_subject_government_and_politics', 'ckan_subject_nature_and_environment']
source_labels = ['ckan_subject_nature_and_environment']
#source_labels = ['ckan_keywords_trends']
conts = {}
jaccs = {}
with open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_pairs.csv', 'r') as f:
    reader = csv.reader(f)
    #edges = list(tuple([rec[1],rec[3]]) for rec in csv.reader(f, delimiter='|'))
    edges = []
    fl = True
    for rec in csv.reader(f, delimiter='|'):
        if fl:
            fl = False
            continue
        edges.append(tuple([rec[1],rec[3]]))
        if rec[1] not in conts:
            conts[rec[1]] = {}
        if rec[1] not in jaccs:
            jaccs[rec[1]] = {}
        conts[rec[1]][rec[3]] = float(rec[4])
        jaccs[rec[1]][rec[3]] = float(rec[6])
        if rec[3] not in conts:
            conts[rec[3]] = {}
        if rec[3] not in jaccs:
            jaccs[rec[3]] = {}
        conts[rec[3]][rec[1]] = float(rec[5])
        jaccs[rec[3]][rec[1]] = float(rec[6])
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
#mems = []
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
        if p != 0.0 and p!= 1.0:
            print(p)
        #sample[labels.index(l)] = 1.0
    if include:
        s = sum(sample)
        for i in range(len(sample)):
            if labels[i] not in samples:
                samples[labels[i]] = []
            samples[labels[i]].append(sample[i]/s)
print("num of samples: %d" % len(samples))
# finding the number of parameters for each label
cards = {}
sizes = {}
for l, ps in samples.items():
    cards[l] = len(set(ps))
    sizes[l] = len(ps)
M = len(table_labels)
g = nx.Graph()
mis = {}
edges = []
# parent
for il1 in range(len(labels)):
    l1 = labels[il1]
    # child
    for il2 in range(il1+1, len(labels)):
        l2 = labels[il2]
        #mi = mutual_info(np.asarray(samples[l1]), np.asarray(samples[l2]), 2)
        mi = mutual_info(np.asarray(samples[l1]), np.asarray(samples[l2]), 100)
        if l1 not in mis:
            mis[l1] = {}
        if l2 not in mis:
            mis[l2] = {}
        mis[l1][l2] = mi
        mis[l2][l1] = mi
# parent
for l1 in labels:
    # child
    for l2 in labels:
        if l1 != l2:
            mi = mis[l1][l2]
            sz1 = ((cards[l2]-1)*(1-cards[l2]))
            sz2 = float(sizes[l2])/sizes[l1]
            bicp = mi
            if bicp == math.inf:
                print("mi: %f and sz1: %f sz2: %f" % (mi, sz1, sz2))
            edges.append((l1, l2, max(bicp, 0.0)))
g.add_weighted_edges_from(edges)
t=nx.maximum_spanning_tree(g)
print("num of edges: %d" % len(t.edges()))
print("num of nodes: %d" % len(t.nodes()))
tree = []
csf = open('mi_nature_tree_new.csv', 'w')
cswriter = csv.writer(csf, delimiter='|', lineterminator='\n', quoting=csv.QUOTE_NONE)
header = ['parent', 'child', 'mi', 'containment', 'jaccard', 'dotproduct', 'card1', 'card2', 'size1', 'size2']
cswriter.writerow(header)
# visualize taxonomy
#taxonomy = gv.Graph(format='pdf')
taxonomy = gv.Digraph(format='pdf')
for e in sorted(t.edges(data=True)):
    if e[2]['weight'] == 0.0:
        print('removing edge: %s   %s' % (e[0],e[1]))
        continue
    if e[0] not in jaccs:
        taxonomy.edge(e[0], e[1], label='overlap=0')
        tree.append([e[0],e[1],e[2]['weight'],0.0, 0.0, np.dot(samples[e[0]], samples[e[1]]), cards[e[0]],cards[e[1]],sizes[e[0]], sizes[e[1]]])
        print('no cont and jacc available for: ')
        print(e)
        print("---------")
        continue
    if e[1] not in jaccs[e[0]]:
        taxonomy.edge(e[0], e[1], label='overlap=0')
        tree.append([e[0],e[1],e[2]['weight'],0.0, 0.0, np.dot(samples[e[0]], samples[e[1]]), cards[e[0]],cards[e[1]],sizes[e[0]], sizes[e[1]]])
        print('no cont and jacc available for: ')
        print(e)
        print("---------")
        continue
    #if jaccs[e[0]][e[1]] == 1.0:
    #    continue
    if conts[e[0]][e[1]] < conts[e[1]][e[0]]:
        #parent -> child
        taxonomy.edge(e[0], e[1])
        tree.append([e[0],e[1],e[2]['weight'],conts[e[1]][e[0]], jaccs[e[1]][e[0]], np.dot(samples[e[0]], samples[e[1]]), cards[e[0]],cards[e[1]],sizes[e[0]],sizes[e[1]]])
        #taxonomy.edge(e[0], e[1], label='(' + str(format(e[2]['weight'], '.2f')) + ',' + str(format(np.dot(samples[e[0]], samples[e[1]]), '.2f')) + ')')
    else:
        taxonomy.edge(e[1], e[0])
        tree.append([e[1],e[0],e[2]['weight'],conts[e[0]][e[1]], jaccs[e[1]][e[0]], np.dot(samples[e[0]], samples[e[1]]), cards[e[0]],cards[e[1]],sizes[e[0]],sizes[e[1]]])
        #taxonomy.edge(e[1], e[0], label='(' + str(format(e[2]['weight'], '.2f')) + ',' + str(format(np.dot(samples[e[0]], samples[e[1]]), '.2f')) + ')')
    print(e)
    print("---------")
cswriter.writerows(tree)
filename = taxonomy.save(filename='/home/fnargesian/FINDOPENDATA_DATASETS/10k/img/mi_natures_new.dot')
print(filename)
