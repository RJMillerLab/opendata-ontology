import org
import org.load as orgl
import org.cluster as orgc
import importlib

def reload():
    importlib.reload(org)
    importlib.reload(orgl)
    importlib.reload(orgc)

import numpy as np
from sklearn.cluster import AgglomerativeClustering

print("Loading domains")
domains = list(orgl.add_ft_vectors(orgl.iter_domains()))

print("Reduce tags")
tags = orgl.reduce_tag_vectors(domains)

print("Done")
