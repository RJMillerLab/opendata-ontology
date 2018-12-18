import networkx as nx
import csv
import graphviz as gv

# finding the lables that are potentially a part of the taxonomy of
#source_labels = ['ckan_tags_society', 'ckan_tags_health', 'ckan_subject_government_and_politics', 'ckan_subject_nature_and_environment']
source_labels = ['ckan_subject_nature_and_environment']
#source_labels = ['ckan_keywords_trends']
conts = {}
jaccs = {}
with open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_pairs.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    edges = []
    for rec in csv.reader(f, delimiter='|'):
        edges.append((rec[1],rec[3]))
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
g = nx.Graph()
gedges = []
# parent
for l1 in labels:
    # child
    for l2 in labels:
        if l1 != l2:
            if l1 in jaccs and l1 in conts:
                if l2 in jaccs[l1] and l2 in conts[l1]:
                    gedges.append((l1, l2, max(conts[l1][l2], conts[l2][l1])/jaccs[l1][l2]))
g.add_weighted_edges_from(gedges)
t=nx.maximum_spanning_tree(g)
print("num of edges: %d" % len(t.edges()))
print("num of nodes: %d" % len(t.nodes()))
tree = []
csf = open('contjacc_nature_tree_new.csv', 'w')
cswriter = csv.writer(csf, delimiter='|', lineterminator='\n', quoting=csv.QUOTE_NONE)
header = ['parent', 'child', 'cont_jacc', 'containment', 'jaccard']
cswriter.writerow(header)
# visualize taxonomy
taxonomy = gv.Digraph(format='pdf')
for e in sorted(t.edges(data=True)):
    if jaccs[e[0]][e[1]] == 1.0:
        continue
    if e[2]['weight'] == 0.0:
        print('removing edge: %s   %s' % (e[0],e[1]))
        continue
    if conts[e[0]][e[1]] < conts[e[1]][e[0]]:
        taxonomy.edge(e[0], e[1])
        tree.append([e[0],e[1],e[2]['weight'],conts[e[1]][e[0]], jaccs[e[1]][e[0]]])
    else:
        taxonomy.edge(e[1], e[0])
        tree.append([e[0],e[1],e[2]['weight'],conts[e[0]][e[1]], jaccs[e[1]][e[0]]])
    print(e)
    print("---------")
cswriter.writerows(tree)
filename = taxonomy.save(filename='/home/fnargesian/FINDOPENDATA_DATASETS/10k/img/contjacc_natures_new.dot')
print(filename)
