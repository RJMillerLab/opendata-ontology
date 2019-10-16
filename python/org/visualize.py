import json
import org.graph as orgg

def save_to_visualize(g, visfilename):

    gnodes = list(g.nodes)
    root = orgg.get_root_plus(g, gnodes)
    print('root')
    print(root)
    d = dict()
    gd = graph_to_dict(g, root, d)
    json.dump(gd, open(visfilename, 'w'))


def graph_to_dict(g, n, d):
    if len(list(g.successors(n))) == 0:
        d = {"name": g.node[n]["sem"], "ID": n, "size": 1}
    else:
        d = dict()
        d["name"] = g.node[n]["sem"]
        dcs = []
        cs = list(g.successors(n))
        for c in cs:
            dcs.append(graph_to_dict(g, c, d))
        d["children"] = dcs
    return d

