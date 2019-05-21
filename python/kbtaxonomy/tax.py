import sqlite3
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
    cursor.execute("SELECT distinct supercat FROM taxonomy where supercat like '%agri%';")
    agri_nodes = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT distinct subcat FROM taxonomy where subcat like '%agri%';")
    agri_nodes += [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT distinct type FROM types where type like '%agri%';")
    agri_nodes += [row[0] for row in cursor.fetchall()]
    agri_nodes = list(set(agri_nodes))
    print('agri_nodes %d' % len(agri_nodes))
    all_agri_nodes = list(agri_nodes)
    agri_nodes = agri_nodes[:20]
    for n in agri_nodes:
        all_agri_nodes.extend(list(nx.descendants(dg, n)))
        all_agri_nodes.extend(list(nx.ancestors(dg, n)))
    all_agri_nodes = list(set(all_agri_nodes))
    print('all_agri_nodes %d' % len(all_agri_nodes))
    return dg.subgraph(all_agri_nodes)

def get_sub_taxonomy():
    g = nx.Graph()
    dg = nx.DiGraph()
    cursor.execute("SELECT subcat, supercat FROM taxonomy;")
    for row in cursor.fetchall():
        g.add_edge(row[0],row[1])
        dg.add_edge(row[1],row[0])
    gls = []
    sgns = []
    #for sg in nx.connected_components(g):
    for sg in nx.strongly_connected_components(dg):
        gls.append(len(sg))
        sgns.append(sg)
        print(get_root(dg.subgraph(sg)))
        print(len(sg))
        if len(sg) > 500:
            print(get_root(dg.subgraph(sg)))
            print('here')
            #return dg.subgraph(sg)
    print(gls)
    print(sum(i > 200 for i in gls))
    return dg.subgraph(sgns[20])

def get_subtax_entities(g):
    type_entity = dict()
    for n in g.nodes():
        cursor.execute("SELECT distinct entity FROM types where type=?;", (n,))
        es = []
        for row in cursor.fetchall():
            es.append(row[0])
        if len(es) > 0:
            type_entity[n] = es
    return type_entity

def get_doms_from_types(g, type_entity):
    domains = []
    tagdomains = dict()
    for t, es in type_entity.items():
        print(t)
        dom = dict()
        dom['name'] = t
        dom['mean'] = emb.get_features(es)
        for p in get_parents_type(g, t):
            dom['tag'] = p
            domains.append(dom)
            if p not in tagdomains:
                tagdomains[p] = []
            tagdomains[p].append(dom)
    tags, vecs = get_tag_vecs(tagdomains)
    return domains, tagdomains, tags, vecs

def get_parents_type(g, t):
    return list(g.predecessors(t))

def get_tag_vecs(tagdomains):
    tags = []
    vecs = []
    for t, ds in tagdomains.items():
        v = np.mean(np.array([d['mean'] for d in ds]), axis=0)
        tags.append(t)
        vecs.append(v)
    return tags, vecs

def get_root(g):
    return [x for x in g.nodes() if g.out_degree(x)>0 and g.in_degree(x)==0]

sg = get_agri_sub_taxonomy()
type_entity = get_subtax_entities(sg)
domains, tagdomains, tags, vecs = get_doms_from_types(sg, type_entity)
json.dump(domains, open('data/agri_domains.json'))
json.dump(tags, open('data/agri_tags.json'))
json.dump(tagdomains, open('data/agri_tagdomains.json'))
json.dump(vecs, open('data/vecs.json'))
print(len(domains))
print(len(tagdomains))
print(len(tags))
print(len(vecs))


