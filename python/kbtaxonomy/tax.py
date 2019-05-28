import sqlite3
import copy
import json
import numpy as np
import emb
import networkx as nx

YAGO_DB = '/home/fnargesian/YAGO/yago.sqlite'
db = sqlite3.connect(YAGO_DB)
cursor = db.cursor()

def get_agri_sub_taxonomy():
    dg = nx.DiGraph()
    cursor.execute("SELECT subcat, supercat FROM taxonomy;")
    for row in cursor.fetchall():
        dg.add_edge(row[1],row[0])
    agri_nodes = []
    cursor.execute("SELECT distinct type FROM types WHERE (type LIKE '%agri%' OR type LIKE '%food%' OR type LIKE '%farm%') AND (type NOT IN (SELECT supercat as type FROM taxonomy)) LIMIT 100;")
    # or random types
    #cursor.execute("SELECT distinct type FROM types ORDER BY RANDOM() 500;")
    agri_types = [row[0] for row in cursor.fetchall()]
    print('dg nodes before types added: %d' % len(dg.nodes))
    agri_type_vecs = dict()
    agri_types_bckup = list(agri_types)
    for t in agri_types_bckup:
        vec = is_good_type(t)
        if len(vec) != 0:
            dg.add_node(t)
            agri_type_vecs[t] = vec
        else:
            agri_types.remove(t)
    print('agri_types: %d' % len(agri_types))
    print('dg nodes after types added: %d' % len(dg.nodes))
    agri_nodes += agri_types
    agri_nodes = list(set(agri_nodes))
    all_agri_nodes = list(agri_nodes)
    nans = []
    for n in agri_nodes:
        ans = list(nx.ancestors(dg, n))
        all_agri_nodes.extend(ans)
        nans.append(len(ans))
    print('min %d max %d avg %d' % (min(nans), max(nans), sum(nans)/len(nans)))
    all_agri_nodes = list(set(all_agri_nodes))
    print('all_agri_nodes after ancestors: %d' % len(all_agri_nodes))
    print('dg nodes: %d' % len(dg.nodes))
    return dg.subgraph(all_agri_nodes), agri_type_vecs

def is_good_type(t):
    cursor.execute("SELECT distinct entity FROM types WHERE type=?;", (t,))
    es = [row[0] for row in cursor.fetchall()]
    if len(es) == 0:
        print('zero es')
        return  []
    v = emb.get_features(es)
    if len(v) == 0:
        print('no emb')
        return []
    return v

def get_doms_from_types(g, agri_types):
    domains = []
    tagdomains = dict()
    tagcompdomains = dict()
    domaintags = dict()
    for t, es in agri_types.items():
        if len(domains) % 50 == 0:
            print('added %d domains' % len(domains))
        dom = dict()
        dom['name'] = t
        dom['mean'] = agri_types[t]
        domains.append(copy.deepcopy(dom))
        parents = get_parents_type(g, t)
        if len(parents) == 0:
            print('no parents')
        for p in parents:
            if dom['name'] not in domaintags:
                domaintags[dom['name']] = []
            domaintags[dom['name']].append(p)
            if p not in tagcompdomains:
                tagcompdomains[p] = []
            tagcompdomains[p].append(dom)
        #g.remove_node(t)
    tags, vecs = get_tag_vecs(tagcompdomains)
    for t, ds in tagcompdomains.items():
        tagdomains[t] = [d['name'] for d in ds]
    return domains, tagdomains, domaintags, tags, vecs, g

def get_parents_type(g, t):
    return [t]
    #return list(g.predecessors(t))

def get_tag_vecs(tagdomains):
    tags = []
    vecs = []
    for t, ds in tagdomains.items():
        v = np.mean(np.array([d['mean'] for d in ds]), axis=0)
        tags.append(t)
        vecs.append(list(v))
    return tags, vecs

def get_leaves(g):
    return [x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)>0]

def get_root(g):
    return [x for x in g.nodes() if g.out_degree(x)>0 and g.in_degree(x)==0]

sg, agri_types = get_agri_sub_taxonomy()
print('num nodes in subgraph: %d' % len(list(sg.nodes)))
print('num edges in subgraph: %d' % len(list(sg.edges)))
print('num roots: %d' % len(get_root(sg)))
print(get_root(sg))
print('num leaves: %d' % len(get_leaves(sg)))
print('DAG: ')
print(nx.is_directed_acyclic_graph(sg))
for x in sg.nodes:
    if sg.out_degree(x)==0 and sg.in_degree(x)==0:
        print('solo node: %d' % x)
print('agri_types: %d' % len(agri_types))
domains, tagdomains, domaintags, tags, vecs, h = get_doms_from_types(sg.copy(), agri_types)
ls  = get_leaves(h)
for t in tags:
    if t not in ls:
        print('wrong tag selected')
print('num roots: %d' % len(get_root(h)))
print(get_root(h))
print('num leaves: %d' % len(get_leaves(h)))
r = get_root(h)[0]
print('choices from root: %d' % len(list(h.successors(r))))
ls = get_leaves(h)
ps = []
for l in ls:
    ps.append(len(nx.shortest_path(h, r, l)))
print('paths: min %d max %d avg %d' % (min(ps), max(ps), sum(ps)/len(ps)))
brs = []
for n in h.nodes:
    if n in ls:
        continue
    brs.append(len(list(h.successors(n))))
print('br factors: min %d max %d avg %d' % (min(brs), max(brs), sum(brs)/len(brs)))
if set(tags) != set(ls):
    print('tags and leaves diff')
json.dump(list(h.edges()), open('data/agri_taxonomy.json', 'w'))
json.dump(domains, open('data/agri_domains.json', 'w'))
json.dump(tags, open('data/agri_tags.json', 'w'))
json.dump(tagdomains, open('data/agri_tagdomains.json', 'w'))
json.dump(domaintags, open('data/agri_domaintags.json', 'w'))
json.dump(vecs, open('data/agri_vecs.json', 'w'))
print(len(domains))
print(len(tagdomains))
print(len(tags))
print(len(vecs))
print('DONE')

