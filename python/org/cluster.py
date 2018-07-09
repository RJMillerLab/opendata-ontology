from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import random

def mk_tag_table(tags):
    keys = sorted(tags.keys())
    X = []
    for k in keys:
        X.append(tags[k]['v'])
    return keys, np.array(X)


def basic_clustering(vecs, n_clusters, linkage, affinity):
    cluster = AgglomerativeClustering(n_clusters=n_clusters,  affinity=affinity, linkage=linkage)
    cluster.fit(vecs)
    return cluster

def kmeans_clustering(tags, vecs, n_branching):
    g = nx.DiGraph()
    cid = 0
    g.add_node(cid)
    g.node[cid]['population'] = vecs
    g.node[cid]['rep'] = np.mean(vecs, axis=0)
    g.node[cid]['tags'] = tags
    tosplit = [cid]
    cid += 1
    for cluster in tosplit:
        if len(g.node[cluster]['population']) > n_branching:
            pop = g.node[cluster]['population']
            ts = g.node[cluster]['tags']
            kmeans = KMeans(n_clusters=n_branching, random_state=random.randint(1,1000)).fit(pop)
            children = get_sub_clusters(kmeans, ts, pop)
            for ch in list(children.values()):
                g.add_node(cid)
                g.node[cid]['population'] = ch['population']
                g.node[cid]['rep'] = np.mean(np.array(ch['population']), axis=0)
                g.node[cid]['tags'] = ch['tags']
                g.add_edge(cluster, cid)
                if len(ch['population']) == 1:
                    g.node[cid]['tag'] = ch['tags'][0]
                else:
                    tosplit.append(cid)
                cid += 1
        else:
            for i in range(len(g.node[cluster]['population'])):
                vec = g.node[cluster]['population'][i]
                g.add_node(cid)
                g.node[cid]['population'] = [vec]
                g.node[cid]['rep'] = vec
                g.node[cid]['tag'] = g.node[cluster]['tags'][i]
                g.add_edge(cluster, cid)
                cid += 1
    return g


def get_sub_clusters(kmeans, tags, vecs):
    subs = dict()
    for i in range(len(kmeans.labels_)):
        c = kmeans.labels_[i]
        if c not in subs:
            subs[c] = {'population': [], 'tags': []}
        subs[c]['population'].append(vecs[i])
        subs[c]['tags'].append(tags[i])
    return subs

