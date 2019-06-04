import sqlite3
import random
import copy
import json
import numpy as np
import emb
import networkx as nx

YAGO_DB = '/home/fnargesian/YAGO/yago.sqlite'
NUM_TABLES = 370
MAX_ATTRS = 20
MIN_ATTRS = 1
MIN_ROWS = 10
db = sqlite3.connect(YAGO_DB)
cursor = db.cursor()

def get_agri_sub_taxonomy():
    dg = nx.DiGraph()
    cursor.execute("SELECT subcat, supercat FROM taxonomy;")
    for row in cursor.fetchall():
        dg.add_edge(row[1],row[0])
    agri_nodes = []
    agri_type_entities = dict()
    cursor.execute("SELECT distinct type FROM types WHERE (type LIKE '%agri%' OR type LIKE '%food%' OR type LIKE '%farm%') AND (type NOT IN (SELECT supercat as type FROM taxonomy)) AND (SELECT COUNT(DISTINCT entity) FROM types t2 where t2.type=types.type) > 10 LIMIT 400;")

    agri_types = [row[0] for row in cursor.fetchall()]
    print('agri types: %d' % len(agri_types))
    print('dg nodes before types added: %d' % len(dg.nodes))
    agri_types_bckup = list(agri_types)
    for t in agri_types_bckup:
        es = get_good_entities(t)
        if len(es) != 0:
            dg.add_node(t)
            agri_type_entities[t] = es
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
    return dg.subgraph(all_agri_nodes), agri_type_entities

def get_good_entities(t):
    cursor.execute("SELECT distinct entity FROM types WHERE type=?;", (t,))
    es = [row[0] for row in cursor.fetchall()]
    if len(es) == 0:
        print('zero es')
        return  []
    es = emb.get_vals_fasttext(es)
    if len(es) == 0:
        print('no emb')
        return []
    return es




def get_topic_vec(es):
    if len(es) == 0:
        print('zero es')
        return  []
    v = emb.get_features(es)
    if len(v) == 0:
        print('no emb')
        return []
    return v



def get_type_topic_vec(t):
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


def get_tables_from_types(agri_type_entities):
    # first assign one attribute to each tag
    tagcompdomains, tagdomains, domaintags = dict(), dict(), dict()
    domains = []
    tables = []
    table_tags = []
    assigned = 0
    agri_types = list(agri_type_entities.keys())
    ats = 0
    while assigned < len(agri_types):
        num_attrs = random.randint(MIN_ATTRS, MAX_ATTRS)
        ats += num_attrs
        tags = agri_types[assigned:(assigned+num_attrs)]
        num_rows = get_num_rows(agri_type_entities, tags)
        table = make_table(tags, agri_type_entities, num_rows)
        tables.append(table)
        table_tags.append(tags)
        assigned += num_attrs
    for j in range(len(tables), NUM_TABLES):
        num_attrs = get_num_attrs_zipfian(MAX_ATTRS)
        ats += num_attrs
        tags = sample_tags_zipfian(agri_types, num_attrs)
        num_rows = get_num_rows(agri_type_entities, tags)
        table = make_table(tags, agri_type_entities, num_rows)
        tables.append(table)
        table_tags.append(tags)

    print('num tables: %d' % len(tables))
    print('num ats: %d' % ats)

    for i in range(len(tables)):
        columns = tables[i]
        tags = table_tags[i]
        table_name = "table_%d" % i
        if len(domains) % 50 == 0:
            print('added %d domains' % len(domains))
        for j in range(len(columns)):
            dom = dict()
            dom['name'] = table_name + '_' + str(j)
            dom['mean'] = get_topic_vec(agri_type_entities[tags[j]])
            domains.append(copy.deepcopy(dom))
            domaintags[dom['name']] = [tags[j]]
            if tags[j] not in tagdomains:
                tagdomains[tags[j]] = []
            if dom['name'] not in tagdomains[tags[j]]:
                tagdomains[tags[j]].append(dom['name'])

            if tags[j] not in tagcompdomains:
                tagcompdomains[tags[j]] = []
            tagcompdomains[tags[j]].append(dom)
    tags, vecs = get_tag_vecs(tagcompdomains)
    return domains, tagdomains, domaintags, tags, vecs

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

def get_num_attrs_zipfian(max_attrs):
    y = np.random.uniform(0.0,1.0)
    x0 = max_attrs
    x1 = 1
    n = -0.9
    # convert uniform to zipfian
    z = int((((x1**(n+1) - x0**(n+1))*y + x0**(n+1))**(1/(n+1))))
    #clip sample ids
    while z < 1 or z > max_attrs:
        u = np.random.uniform(0.0, 1.0)
        z = int((((x1**(n+1) - x0**(n+1))*u + x0**(n+1))**(1/(n+1))))
    return z

def sample_tags_zipfian(tags, num):
    y = np.random.uniform(0.0,1.0,num)
    x0 = len(tags)
    x1 = 1
    n = -0.9
    z = (((x1**(n+1) - x0**(n+1))*y + x0**(n+1))**(1/(n+1))).round().astype(int)
    for i in range(len(z)):
        while z[i] < 1 or z[i] > len(tags):
            u = np.random.uniform(0.0, 1.0, 1)
            r = (((x1**(n+1) - x0**(n+1))*u + x0**(n+1))**(1/(n+1))).round().astype(int)
            z[i] = r
    z = [x-1 for x in z]
    return list(np.asarray(tags)[z])

def make_table(tags, type_entities, size):
    table = []
    for w in tags:
        table.append(make_column(size, type_entities[w]))
    return table

def make_column(size, es):
    return random.sample(es, size)

def get_num_rows(tag_entities, tags):
    max_rows = min([len(tag_entities[t]) for t in tags])
    return random.randint(10, max_rows)



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
domains, tagdomains, domaintags, tags, vecs = get_tables_from_types(agri_types)
h = sg
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
json.dump(list(h.edges()), open('yagocloud_data/agri_taxonomy.json', 'w'))
json.dump(domains, open('yagocloud_data/agri_domains.json', 'w'))
json.dump(tags, open('yagocloud_data/agri_tags.json', 'w'))
json.dump(tagdomains, open('yagocloud_data/agri_tagdomains.json', 'w'))
json.dump(domaintags, open('yagocloud_data/agri_domaintags.json', 'w'))
json.dump(vecs, open('yagocloud_data/agri_vecs.json', 'w'))
print(len(domains))
print(len(tagdomains))
print(len(tags))
print(len(vecs))
print('DONE')

