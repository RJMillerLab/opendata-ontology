import json
import copy
import numpy as np
import networkx as nx

def trim(tags, domains):
    tagdomains_tr = copy.deepcopy(tagdomains)
    domaintags = dict()
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e[0], e[1])
    print('nodes %d edges %d before trimming' % (len(g.nodes), len(g.edges)))
    leaves = get_leaves(g)
    print('leaves: %d' % len(leaves))
    print('tags: %d' % len(tags))
    for l in leaves:
        if l not in tags:
            print('%s not in tags' % l)
        for p in g.predecessors(l):
            for gp in g.predecessors(p):
                if gp not in tagdomains_tr:
                    tagdomains_tr[p] = []
                tagdomains_tr[p].extend(tagdomains_tr[l])
                tagdomains_tr[p] = list(set(tagdomains_tr[l]))
        del tagdomains_tr[l]
        g.remove_node(l)
    tags, vecs = get_tag_vecs(tagdomains_tr, domains)
    for t, ds in tagdomains_tr.items():
        for d in ds:
            if d not in domaintags:
                domaintags[d] = []
            domaintags[d].append(t)
    print('nodes %d edges %d after trimming' % (len(g.nodes), len(g.edges)))
    print('tags: %d' % len(tags))
    return tagdomains_tr, domaintags, tags, vecs, g.copy()


def get_leaves(g):
    return [x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)>0]


def get_tag_vecs(tds, domains):
    domain_index = dict()
    for dom in domains:
        if dom['name'] not in domain_index:
            domain_index[dom['name']] = dom['mean']
    tags = []
    vecs = []
    for t, ds in tds.items():
        v = np.mean(np.array([domain_index[d] for d in ds]), axis=0)
        tags.append(t)
        vecs.append(list(v))
    return tags, vecs



edges = json.load(open('yagocloud_data/agri_taxonomy.json'))
tags = json.load(open('yagocloud_data/agri_tags.json'))
tagdomains = json.load(open('yagocloud_data/agri_tagdomains.json'))
domains = json.load(open('yagocloud_data/agri_domains.json'))

tagdomains_tr, domaintags, tags, vecs, h = trim(tags, domains)
while(len(h.nodes)>300):
    print('nodeS: %d' % len(h.nodes))
    tagdomains_tr, domaintags, tags, vecs, h = trim(tags, domains)

json.dump(tags, open('yagocloud_data/agri_tags_trim.json', 'w'))
json.dump(tagdomains_tr, open('yagocloud_data/agri_tagdomains_trim.json', 'w'))
json.dump(domaintags, open('yagocloud_data/agri_domaintags_trim.json', 'w'))
json.dump(vecs, open('yagocloud_data/agri_vecs_trim.json', 'w'))

