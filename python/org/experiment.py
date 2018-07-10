import org.cluster as orgc
import org.graph as orgg
import org.hierarchy as orgh

def get_tag_ranks_basic(tags, vecs, params, domains):
    results = []
    for ncs in params['num_clusters']:
        for m in params['measures']:
            error = 0
            n_clusters = ncs
            linkage = m[0]
            affinity = m[1]
            print('linkage: %s and affinity: %s and n_clusters: %d' %(linkage, affinity, ncs))
            gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, n_clusters, linkage, affinity), vecs, tags), vecs)
            tag_dists, tag_ranks = orgh.get_reachability_probs(gp, domains)
            print("tag ranks: {}".format(tag_ranks))
            print("Computing reachability probs")
            error = sum(tag_ranks.values())
            print('error: %d' % error)
            rs = {'n_clusters':  n_clusters, 'linkage': linkage, 'affinity': affinity, 'tag_dists': tag_dists, 'tag_ranks': tag_ranks, 'rank_error': error}
            results.append(rs)
    return results


def get_tag_ranks_kmeans(tags, vecs, params, domains):
    results = []
    for ncs in params['n_branches']:
        #error = 0
        n_branches = ncs
        gp = orgc.kmeans_clustering(tags, vecs, n_branches)
        rs = orgh.evaluate(gp, domains)
        #orgh.get_reachability_probs(gp, domains)
        #print("tag ranks: {}".format(tag_ranks))
        #print("Computing reachability probs")
        #error = sum(tag_ranks.values())
        #print('error: %d' % error)
        #rs = {'n_branches':  n_branches, 'tag_dists': tag_dists, 'tag_ranks': tag_ranks, 'rank_error': error}
        results.append(rs)
    return results

