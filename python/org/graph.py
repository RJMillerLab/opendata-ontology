import networkx as nx

def cluster_to_graph(cluster, vecs, tags):
    n_leaves = len(vecs)
    edges = [(n_leaves+i, child) for i in range(len(cluster.children_)) for child in cluster.children_[i]]
    g=nx.DiGraph()
    g.add_edges_from(edges)
    for n in get_leaves(g):
        g.node[n]['tag'] = tags[n]
    return g


def get_flat_cluster_graph(tags):
    root = len(tags)
    edges = []
    for i in range(len(tags)):
        edges.append((root, i))
    g=nx.DiGraph()
    g.add_edges_from(edges)
    for n in get_leaves(g):
        g.node[n]['tag'] = tags[n]
    return g


def get_leaves(g):
    return set([x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)>0])


def get_root(g):
    return [x for x in g.nodes() if g.out_degree(x)>0 and g.in_degree(x)==0][0]


def get_siblings(g, n, p):
    siblings = []
    for p in list(g.predecessors(n)):
        for s in list(g.successors(p)):
            siblings.append(s)
    return siblings


def level_up(g, nodes):
    ups = []
    for n in nodes:
        if n not in g:
            continue
        ps = list(g.predecessors(n))
        for s in ps:
            if s not in ups and s not in nodes:
                ups.append(s)
    return ups


def height(g):
    ds = []
    r = get_root(g)
    for l in get_leaves(g):
        ds.append(nx.shortest_path_length(g,source=r,target=l))
    print('heights: {}'.format(ds))
    print('min height: %d  max height: %d' % (min(ds), max(ds)))




