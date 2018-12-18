import networkx as nx
import numpy as np
import csv
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

labels = ['l1', 'l2', 'l3']
samples = {}
samples['l1'] = [0.1,1,1,1,1,0,0.7,0,0,0]
samples['l2'] = [0.1,1,1,1,1,0,0.7,0,0,0]
samples['l3'] = [1,1,0,0,0,1,0,1,1,1]
cards = {}
sizes = {}
cards['l1'] = 10#4
cards['l2'] = 10#4
cards['l3'] = 10#2
sizes['l1'] = 6
sizes['l2'] = 4
sizes['l3'] = 2
M = 10
g = nx.Graph()
mis = {}
edges = []
# parent
for il1 in range(len(labels)):
    l1 = labels[il1]
    # child
    for il2 in range(il1+1, len(labels)):
        l2 = labels[il2]
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
            #sz1 = np.log2((cards[l2]*cards[l2]-cards[l2]+cards[l1]-1)/float(cards[l1]+cards[l2]-2))
            sz1 = ((cards[l2]-1)*(1-cards[l2]))
            sz2 = float(sizes[l2])/sizes[l1]
            #bicp = M * mi - np.dot(np.asarray(samples[l1]), np.asarray(samples[l2]))#- (np.log2(M)/2.0)*(sz2+np.abs(sz1))
            bicp = mi
            #print("%s - %s mi: %f and sz1: %f sz2: %f log: %f = %f -> %f" % (l1, l2, mi, sz1, sz2, np.log2(M), (np.log2(M)/2.0)*(sz1 * sz2), bicp))
            #print("%s - %s: %f" % (l1, l2, bic))
            #if bicp == math.inf:
            print("%s %s mi: %f" % (l1, l2, mi))
            edges.append((l1, l2, max(bicp, 0.0)))
g.add_weighted_edges_from(edges)
print(len(g.nodes()))
print(len(g.edges()))
t=nx.maximum_spanning_tree(g)
print("num of edges: %d" % len(t.edges()))
print("num of nodes: %d" % len(t.nodes()))
print(sizes)
print(cards)
tree = []
csf = open('test_tree.csv', 'w')
cswriter = csv.writer(csf, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)
header = ['label1', 'label2', 'mi', 'dotprod', 'card1', 'card2', 'size1', 'size2']
cswriter.writerow(header)
for e in sorted(t.edges(data=True)):
    if e[2]['weight'] != 0.0:
        tree.append([e[0],e[1],e[2]['weight'],np.dot(samples[e[0]], samples[e[1]]), cards[e[0]],cards[e[1]],sizes[e[0]],sizes[e[1]]])
        print(e)
        print("%f" % (np.dot(samples[e[0]], samples[e[1]])))
        print("---------")
cswriter.writerows(tree)
taxonomy = gv.Graph(format='pdf')
print(list(g.edges()))
for e in sorted(t.edges(data=True)):
    c = 'blue'
    dp = np.dot(samples[e[0]], samples[e[1]])
    print(dp)
    if dp < 0.5:
        c = 'red'
    taxonomy.edge(e[0], e[1], label='(' + str(format(e[2]['weight'], '.2f')) + ',' + str(format(np.dot(samples[e[0]], samples[e[1]]), '.2f')) + ')', color=c)
filename = taxonomy.save(filename='/home/fnargesian/FINDOPENDATA_DATASETS/10k/img/test.dot')
print(filename)
#filename = taxonomy.render(filename='/home/fnargesian/FINDOPENDATA_DATASETS/10k/img/natures.viz')

