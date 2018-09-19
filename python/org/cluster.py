from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import random
import org.hierarchy as orgh
import operator
import skfuzzy as fuzz
from sklearn.neighbors import kneighbors_graph
import datetime


def mk_tag_table(tags):
    keys = sorted(tags.keys())
    X = []
    for k in keys:
        X.append(tags[k]['v'])
    return keys, np.array(X)


def basic_clustering(vecs, n_clusters, linkage, affinity):
    knn_graph = kneighbors_graph(vecs, min(50, len(vecs)-1), include_self=False)
    cluster = AgglomerativeClustering(n_clusters=n_clusters,  affinity=affinity, linkage=linkage, memory='/home/fnargesian/tmp', connectivity=knn_graph)
    cluster.fit(vecs)
    return cluster


def basic_plus(vecs, n_clusters, linkage, affinity):
    knn_graph = kneighbors_graph(vecs, min(50, len(vecs)-1) , include_self=False)
    t0 = datetime.datetime.now()
    cluster = AgglomerativeClustering(n_clusters=n_clusters,  affinity=affinity, linkage=linkage)
    elapsed_time = datetime.datetime.now() - t0
    print(elapsed_time)


    for connectivity in (None, knn_graph):
        t0 = datetime.datetime.now()
        cluster = AgglomerativeClustering(n_clusters=n_clusters,  affinity=affinity, linkage=linkage, connectivity=connectivity)
        elapsed_time = datetime.datetime.now() - t0
        print(elapsed_time)

    #ys = cluster.fit_predict(vecs[:-100])
    #print('comps: %d' % cluster.n_components_)
    #print(ys)
    #print('children: %d ys: %d' % (len(cluster.children_), len(ys)))
    #n_leaves = len(vecs)
    #edges = [(n_leaves+i, child) for i in range(len(cluster.children_)) for child in cluster.children_[i]]
    #print(edges)
    #print(len(cluster.children_))



    return cluster




def equi_depth_partition(vecs, tags, n_cluster):
    n_branching = int(len(tags)/n_cluster)
    # lookup map of tag names and ids
    tmap = dict()
    for i in range(len(tags)):
        tmap[tags[i]] = i
    sims = dict()
    tag_closeness = dict()
    for i in range(len(tags)):
        sims[tags[i]] = dict()
        tag_closeness[tags[i]] = 0.0
        #for j in range(i+1,len(tags)):
        for j in range(len(tags)):
            if j == i:
                continue
            if tags[j] not in sims:
                sims[tags[j]] = dict()
                tag_closeness[tags[j]] = 0.0
            s = orgh.get_transition_sim(vecs[i], vecs[j])
            sims[tags[i]][tags[j]] = s
            #sims[tags[j]][tags[i]] = s
            tag_closeness[tags[i]] += s
            #tag_closeness[tags[j]] += s
    order = sorted(tag_closeness.items(), key=operator.itemgetter(1))
    clustered = []
    clusters = dict()
    for o in order:
        t = o[0]
        if t in clustered:
            continue
        neighbors = sorted(sims[t].items(), key=operator.itemgetter(1), reverse=True)
        clusters[t] = [t]
        clustered.append(t)
        for n in neighbors:
            if n[0] not in clustered and len(clusters[t])<n_branching:
                clusters[t].append(n[0])
                clustered.append(n[0])
    return clusters


def balanced_clustering(tags, vecs, n_branching):
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
            clusters = equi_depth_partition(g.node[cluster]['population'], g.node[cluster]['tags'], n_branching)
            children = get_sub_clusters(clusters, ts, pop)
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




def kmeans_clustering(tags, vecs, n_branching):
    g = nx.DiGraph()
    cid = 0
    g.add_node(cid)
    g.node[cid]['population'] = list(vecs)
    g.node[cid]['rep'] = np.mean(vecs, axis=0)
    g.node[cid]['tags'] = tags
    tosplit = [cid]
    cid += 1
    for cluster in tosplit:
        if len(g.node[cluster]['population']) > n_branching:
            pop = g.node[cluster]['population']
            ts = g.node[cluster]['tags']
            kmeans = KMeans(n_clusters=n_branching, random_state=random.randint(1,1000)).fit(pop)
            children = get_kmeans_sub_clusters(kmeans, ts, pop)
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


def get_kmeans_sub_clusters(kmeans, tags, vecs):
    subs = dict()
    for i in range(len(kmeans.labels_)):
        c = kmeans.labels_[i]
        if c not in subs:
            subs[c] = {'population': [], 'tags': []}
        subs[c]['population'].append(vecs[i])
        subs[c]['tags'].append(tags[i])
    return subs


def get_sub_clusters(clusters, tags, vecs):
    subs = dict()
    j = 0
    for s, ts in clusters.items():
        subs[j] = {'population': [], 'tags': []}
        for t in ts:
            i = tags.index(t)
            subs[j]['population'].append(vecs[i])
            subs[j]['tags'].append(tags[i])
        j += 1
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
    children = get_kmeans_sub_clusters(kmeans, tags, vecs)
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



def complete_kary_cluster(tags, vecs, n_cluster):
    n_branching = int(len(tags)/n_cluster)
    # lookup map of tag names and ids
    tmap = dict()
    for i in range(len(tags)):
        tmap[tags[i]] = i
    sims = dict()
    tag_closeness = dict()
    for i in range(len(tags)):
        sims[tags[i]] = dict()
        tag_closeness[tags[i]] = 0.0
        for j in range(i+1,len(tags)):
            s = orgh.get_transition_sim(vecs[i], vecs[j])
            sims[tags[i]][tags[j]] = s
            tag_closeness[tags[i]] += s
    order = sorted(tag_closeness.items(), key=operator.itemgetter(1))
    clustered = []
    clusters = dict()
    for o in order:
        t = o[0]
        if t in clustered:
            continue
        neighbors = sorted(sims[t].items(), key=operator.itemgetter(1), reverse=True)
        clusters[t] = [t]
        clustered.append(t)
        for n in neighbors:
            if n[0] not in clustered and len(clusters[t])<n_branching:
                clusters[t].append(n[0])
                clustered.append(n[0])
    g = nx.DiGraph()
    # root
    rid = 0
    g.add_node(rid)
    g.node[rid]['population'] = vecs
    g.node[rid]['rep'] = np.mean(vecs, axis=0)
    g.node[rid]['tags'] = tags
    cid = rid
    for c in clusters.values():
        if len(c) < n_branching:
            continue
        cid += 1
        pop = []
        ts = []
        tid = cid
        g.add_node(cid)
        for t in c:
            tid += 1
            pop.append(vecs[tmap[t]])
            ts.append(tags[tmap[t]])
            g.add_node(tid)
            g.node[tid]['population'] = [vecs[tmap[t]]]
            g.node[tid]['rep'] = vecs[tmap[t]]
            g.node[tid]['tag'] = t
            g.add_edge(cid, tid)
        g.node[cid]['population'] = pop
        g.node[cid]['rep'] = np.mean(np.array(pop), axis=0)
        g.node[cid]['tags'] = ts
        g.add_edge(rid, cid)
        cid = tid
    return g

def cmeans_clustering(tags, vecs):
    tags = np.array(tags)
    vecs = np.array(vecs)

    for ncenters in range(2,8):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(vecs.T, ncenters, 2, error=0.005, maxiter=1000, init=None)
        print('%d: fpc: %f' % (ncenters, fpc))
    dim_membership = np.argmax(u, axis=0)
    dims = dict()
    for i in range(ncenters):
        inx = dim_membership == i
        c = i
        if c not in dims:
            dims[c] = {'population': [], 'tags': []}
            dims[c]['population'] = list(vecs[inx])
            dims[c]['tags'] = list(tags[inx])
    return dims

