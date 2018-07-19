import org.cluster as orgc
import org.graph as orgg
import org.hierarchy as orgh

def get_tag_ranks_basic(tags, vecs, params, domains):
    results = []
    for ncs in params['num_clusters']:
        for m in params['measures']:
            n_clusters = ncs
            linkage = m[0]
            affinity = m[1]
            #print('linkage: %s and affinity: %s and n_clusters: %d' %(linkage, affinity, ncs))
            gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, n_clusters, linkage, affinity), vecs, tags), vecs)
            rs = orgh.evaluate(gp, domains)
            results.append(rs)
    return results


def get_tag_ranks_kmeans(tags, vecs, params, domains):
    results = []
    for ncs in params['n_branches']:
        n_branches = ncs
        gp = orgc.kmeans_clustering(tags, vecs, n_branches)
        rs = orgh.evaluate(gp, domains)
        results.append(rs)
    return results


def evaluate_likelihood(tags, vecs, params, domains):
    results = []
    for ncs in params['n_branches']:
        n_branches = ncs
        gp = orgc.kmeans_clustering(tags, vecs, n_branches)
        rs = orgh.log_likelihood(gp, domains)
        results.append(rs)
    return results


def evaluate_state_probs(tags, vecs, params, domains):
    print('evaluate_state_probs')
    results = []
    for ncs in params['n_branches']:
        n_branches = ncs
        g = orgc.kmeans_clustering(tags, vecs, n_branches)
        state_probs, h = orgh.get_state_probs(g, domains)
        print(state_probs)
    results.append(state_probs)


