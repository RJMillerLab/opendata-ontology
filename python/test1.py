import org
import org.load as orgl
import org.cluster as orgc
import org.graph as orgg
import importlib

def reload():
    importlib.reload(org)
    importlib.reload(orgl)
    importlib.reload(orgc)
    importlib.reload(orgg)


print("Loading domains")
domains = list(orgl.add_ft_vectors(orgl.iter_domains()))

print("Reduce tags")
tags = orgl.reduce_tag_vectors(domains)

print("Basic clustering")
keys, vecs = orgc.mk_tag_table(tags)
num_clusters = 20
c = orgc.basic_clustering(vecs, num_clusters)

print("Cluster to graph")
g = orgg.cluster_to_graph(c, vecs, keys)
gp = orgg.add_node_vecs(g, vecs)

print("Computing reachability probs")
tag_ranks = dict()
tag_dists = dict()
for domain in domains[:10]:
    tag_dist = orgg.get_tag_probs(gp, domain)
    tag_dists[domain['tag']] = tag_dist
    tag_ranks[domain['tag']] = [i for i in range(len(tag_dist)) if tag_dist[i][0]==domain['tag']][0]
print("tag ranks: {}".format(tag_ranks))

print("Done")
