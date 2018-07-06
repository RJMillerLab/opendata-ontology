import networkx as nx
import operator
from scipy import spatial
import numpy as np
import math


def cluster_to_graph(cluster, vecs, tags):
    n_leaves = len(vecs)
    edges = [(n_leaves+i, child) for i in range(len(cluster.children_)) for child in cluster.children_[i]]
    g=nx.DiGraph()
    g.add_edges_from(edges)
    for n in get_leaves(g):
        g.node[n]['tag'] = tags[n]
    return g


def add_node_vecs(g, vecs):
    leaves = get_leaves(g)
    for n in g.nodes:
        if n not in leaves:
            g.node[n]['vec'] = np.mean(vecs[np.array(list(leaves.intersection(nx.descendants(g,n))))], axis=0)
        else:
            g.node[n]['vec'] = vecs[n]
    return g


def get_tag_probs(g, domain):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_node_probs(gp)
    tags = dict()
    for n in get_leaves(gpp):
        tags[gpp.node[n]['tag']] = gpp.node[n]['reach_prob']
    tag_sorted = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
    return tag_sorted


def get_domain_edge_probs(g, domain):
    gd = g.copy()
    # computing sims
    for e in gd.edges:
        p, ch = e[0], e[1]
        gd[p][ch]['trans_sim'] = get_transition_prob(gd.node[ch]['vec'], domain['mean'])
    # computing softmax prob
    for e in gd.edges:
        p, ch = e[0], e[1]
        d = sum([math.exp(gd[p][s]['trans_sim']) for s in gd.successors(p)])
        gd[p][ch]['trans_prob'] = math.exp(gd[p][ch]['trans_sim'])/d
    return gd


def get_node_probs(g):
    gd = g.copy()
    for n in gd.node:
        if n == get_root(gd):
            gd.node[n]['reach_prob'] = 1.0
        else:
            gd.node[n]['reach_prob'] = 0.0
    top = list(nx.topological_sort(gd))
    for p in top:
        for ch in gd.successors(p):
            gd.node[ch]['reach_prob'] += gd.node[p]['reach_prob']*gd[p][ch]['trans_sim']
    return gd


def get_leaves(g):
    return set([x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)>0])


def get_root(g):
    return [x for x in g.nodes() if g.out_degree(x)>0 and g.in_degree(x)==0][0]


def get_transition_prob(vec1, vec2):
    return 1.0 - spatial.distance.cosine(vec1, vec2)


def get_siblings(g, n, p):
    siblings = []
    for p in g.predecessors(n):
        for s in g.successors(p):
            siblings.append(s)
    return siblings


