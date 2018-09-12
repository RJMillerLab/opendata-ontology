import networkx as nx
#import multiprocessing
#from itertools import repeat
import sqlite3
import numpy as np
#from networkx.readwrite import json_graph
import math
import graph as orgg


g = nx.DiGraph()

def init():
    global g
    g.add_edges_from([(1,2), (1,3), (2,4)])
    g.node[1]['name'] = 'a'
    g.node[1]['v'] = [1,0.2]
    return g

def f(s, t, u):
    print(u+t)
    g.add_edges_from([(2,5)])
    g.node[3]['y']=s
    print(g.node[3])
    return s+'3', g.node[3]['y']


def read_emb():
    DB_FILE = '/home/kenpu/clones/tagcloud-nlp-generator/ft.sqlite3'
    db = sqlite3.connect(DB_FILE)
    c = db.cursor()
    sql = "select word, vec from wv where word = 'test'"
    c.execute(sql)
    dt = np.dtype(np.float32)
    dt = dt.newbyteorder('<')
    features = []
    for row in c.fetchall():
        print('buffer size: %d' % len(row[1]))
        emb_blob = row[1]
        print(row[1][len(row[1])-1])
        emb_vec = np.frombuffer(emb_blob, dtype=dt)
        if np.isnan(emb_vec).any():
            print('found nan')
        features.append(emb_vec)
    print(features)

def transprob():
    a = [0.01, 0.01, 0.9]
    b = [0.001 for i in range(20)] + [0.9]
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    s6 = 0.0
    for x in a:
        s1 += math.exp(x)
        s2 += math.exp(x*(100/3.0))
        s3 += math.exp(x*3)
    for x in b:
        s4 += math.exp(x)
        s5 += math.exp(x*(100/21.0))
        s6 += math.exp(x*21)
    print('s1: %f s4: %f' % (s1, s4))
    print('s2: %f s5: %f' % (s2, s5))
    print('s3: %f s6: %f' % (s3, s6))
    for x in a:
        print('%f %f (*)%f  (/)%f' % (x, (math.exp(x)/s1), (math.exp(x*1.0/3)/s2), (math.exp(x*3)/s3)))
    print('-------')
    for x in b:
        print('%f %f (*)%f  (/)%f' % (x, (math.exp(x)/s4), (math.exp(x*1.0/21)/s5), (math.exp(x*21)/s6)))


#g = init()
#print(g.node[3])
#ss = ['a', 'b', 'c', '4', 'u']
#pool = multiprocessing.Pool(3)
#results = pool.starmap(f, zip(ss, repeat('2'), repeat('t')))
#results = pool.map(f, ss)
#f('w', 'a','a')
#print(g.node[3])
#read_emb()
#nx.write_gpickle(g, "test.gpickle")
#h = nx.read_gpickle("test.gpickle")
#print(h.nodes)
#print(h.edges)
#print(h.node[1])
#print(json_graph.dumps(g))
#with open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_agg.json', 'w') as ff:
#    ff.write(json_graph.dumps(g))

#transprob()
gs = []
g = nx.DiGraph()
g.add_edges_from([(1,2), (1,3), (2,4)])
g.node[2]['name'] = 'ew'
g.node[2]['arr'] = [1.0, 0.9]
g.node[4]['name'] = 'fn'
g.node[4]['arr'] = [1.1, 0.99]
gs.append(g.copy())
gs.append(g.copy())
gs.append(g.copy())
h = orgg.merge_graphs(gs)
print(h.nodes)
print(h.edges)
print(g.node[2]['name'])
print(h.node[2]['name'])
print(h.node[2]['arr'])
print(h.node[8]['name'])
print(h.node[8]['arr'])

print('done')

