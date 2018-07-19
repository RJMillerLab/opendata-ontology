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
                #g.node[cid]['cov'] = np.cov(np.transpose(g.node[cid]['population']))
                #g.node[cid]['mean'] = g.node[cid]['rep']
                #g.node[cid]['det'] = linalg.det(g.node[cid]['cov'])
                #print(g.node[cid]['det'])
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


def twolevel_kmeans_clustering(tags, vecs, n_branching):
    g = nx.DiGraph()
    # root
    rid = 0
    g.add_node(rid)
    g.node[rid]['population'] = vecs
    g.node[rid]['rep'] = np.mean(vecs, axis=0)
    g.node[rid]['tags'] = tags
    kmeans = KMeans(n_clusters=n_branching, random_state=random.randint(1,1000)).fit(vecs)
    children = get_sub_clusters(kmeans, tags, vecs)
    cid = rid
    for ch in list(children.values()):
        cid += 1
        g.add_node(cid)
        g.node[cid]['population'] = ch['population']
        g.node[cid]['rep'] = np.mean(np.array(ch['population']), axis=0)
        g.node[cid]['tags'] = ch['tags']
        g.add_edge(rid, cid)
        # adding singlton nodes
        lid = cid
        for i in range(len(g.node[cid]['population'])):
            lid += 1
            vec = g.node[cid]['population'][i]
            g.add_node(lid)
            g.node[lid]['population'] = [vec]
            g.node[lid]['rep'] = vec
            g.node[lid]['tag'] = g.node[cid]['tags'][i]
            g.add_edge(cid, lid)
        cid = lid
    return g































