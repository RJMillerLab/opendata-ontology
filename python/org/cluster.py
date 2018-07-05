from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import numpy as np

def mk_tag_table(tags):
    keys = sorted(tags.keys())
    X = []
    for k in keys:
        X.append(tags[k]['v'])
    return keys, np.array(X)



