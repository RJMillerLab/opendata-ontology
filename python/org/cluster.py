from sklearn.cluster import AgglomerativeClustering
import numpy as np

def mk_tag_table(tags):
    keys = sorted(tags.keys())
    X = []
    for k in keys:
        X.append(tags[k]['v'])
    return keys, np.array(X)


def basic_clustering(vecs, num_clusters):
    cluster = AgglomerativeClustering(n_clusters=num_clusters,  affinity='cosine', linkage='average')
    cluster.fit(vecs)
    return cluster


