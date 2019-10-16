import json
import csv
import graph as orgg
import networkx as nx

node_ids = dict()

def save_to_visualize(visfilename):
    global node_ids

    g = nx.DiGraph()
    g.add_weighted_edges_from([("1","2", 0.1), ("1","3",0.2), ("2","4",0.4), ("2","5",0.4), ("3","6",0.4), ("3", "7", 0.5), ("3", "8", 56), ("8", "9", 1), ("4", "10", 10), ("4", "11", 10)])
    for n in g.nodes:
        g.node[n]["sem"] = "test"
    gnodes = list(g.nodes)
    root = orgg.get_root_plus(g, gnodes)
    print('root')
    print(root)
    d = dict()
    gd = graph_to_dict(g, root, d, "0")
    json.dump(gd, open(visfilename, 'w'))

    with open('names.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name in list(node_ids.keys()):
            writer.writerow({'name': name, 'ID': node_ids[name]})


def graph_to_dict(g, n, d, index):
    if len(list(g.successors(n))) == 0:
        #d = {"name": g.node[n]["sem"], "ID": index, "size": 10}
        d = {"name": n, "ID": index, "size": 10}
    else:
        d = dict()
        #d["name"] = g.node[n]["sem"]
        d = {"name": n}
        if index != "0":
            node_ids[n] = index
        dcs = []
        cs = list(g.successors(n))
        for i in range(len(cs)):
            c = cs[i]
            if index == "0":
                dcs.append(graph_to_dict(g, c, d, str(i+1)))
            else:
                dcs.append(graph_to_dict(g, c, d, index+"."+str(i+1)))
        d["children"] = dcs
    return d

save_to_visualize('vg.json')
