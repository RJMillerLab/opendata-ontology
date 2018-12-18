import networkx as nx

g = nx.Graph()
edges = [(1,2),(2,3),(2,4),(3,6),(9,3),(9,12),(12,13),(12,14)]
g.add_edges_from(edges)
roots = [1,9]
label_tables = {}
table_labels = {}
label_tables[1] = [3,2]
label_tables[4] = [5,10]
label_tables[3] = [6,11]
label_tables[6] = [7,8]
table_labels[5] = [4]
table_labels[10] = [4]
table_labels[3] = [1]
table_labels[7] = [6]
table_labels[8] = [6]
mdls = 0
for t, ls in table_labels.items():
    mdl = 0
    for l in ls:
        for r in roots:
            mdl += len(nx.shortest_path(g, source=r, target=l))
            mdl += len(label_tables[l])-1
    print("t: %d min desc: %d" % (t, mdl))
    mdls += mdl
print(mdls)
