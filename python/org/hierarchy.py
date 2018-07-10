import networkx as nx
import org.graph as orgg
import operator
from scipy import spatial
import numpy as np
import math

def get_reachability_probs(gp, domains):
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    for domain in domains:
        tags = get_tag_probs(gp, domain)
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['name']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                success_probs[domain['name']] = tag_dist[i][1]
    return tag_dists, tag_ranks, success_probs


def add_node_vecs(g, vecs):
    leaves = orgg.get_leaves(g)
    for n in g.nodes:
        if n not in leaves:
            node_vecs = vecs[np.array(list(leaves.intersection(nx.descendants(g,n))))]
            g.node[n]['population'] = node_vecs
            g.node[n]['rep'] = np.mean(node_vecs, axis=0)
        else:
            g.node[n]['rep'] = vecs[n]
            g.node[n]['population'] = [vecs[n]]
    return g


def get_tag_probs(g, domain):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_node_probs(gp)
    tags = dict()
    for n in orgg.get_leaves(gpp):
        tags[gpp.node[n]['tag']] = gpp.node[n]['reach_prob']
    return tags


def get_domain_edge_probs(g, domain):
    gd = g.copy()
    # computing sims
    for e in gd.edges:
        p, ch = e[0], e[1]
        gd[p][ch]['trans_sim'] = get_transition_prob(gd.node[ch]['rep'], domain['mean'])
        #gd[p][ch]['trans_sim'] = get_transition_prob_plus(domain['mean'], gd.node[ch]['population'])
    # computing softmax prob
    for e in gd.edges:
        p, ch = e[0], e[1]
        d = sum([math.exp(gd[p][s]['trans_sim']) for s in gd.successors(p)])
        gd[p][ch]['trans_prob'] = math.exp(gd[p][ch]['trans_sim'])/d
    return gd


def get_node_probs(g):
    gd = g.copy()
    for n in gd.node:
        if n == orgg.get_root(gd):
            gd.node[n]['reach_prob'] = 1.0
        else:
            gd.node[n]['reach_prob'] = 0.0
    top = list(nx.topological_sort(gd))
    for p in top:
        for ch in gd.successors(p):
            gd.node[ch]['reach_prob'] += gd.node[p]['reach_prob']*gd[p][ch]['trans_prob']
    for n in gd.nodes:
        if gd.node[n]['reach_prob'] == 0.0:
            print('0.0 0.0')
    return gd


def get_transition_prob(vec1, vec2):
    return 1.0 - spatial.distance.cosine(vec1, vec2)

def get_transition_prob_plus(vec1, vecs2):
    s = 0.0
    for vec2 in vecs2:
        s += (1.0 - spatial.distance.cosine(vec1, vec2))
    return s/float(len(vecs2))

def evaluate(g, domains):
    error = 0
    tag_dists, tag_ranks, success_probs = get_reachability_probs(g, domains)
    print('tag ranks: {}'.format(tag_ranks))
    print('Computing reachability probs')
    error = sum(tag_ranks.values()) - len(domains)
    print('error: %d' % error)
    print('domain search success probs: ')
    print(success_probs)
    print('hierarchy success prob: %f' % (sum(list(success_probs.values()))/float(len(success_probs))))
    results = {'tag_dists': tag_dists, 'tag_ranks': tag_ranks, 'success_probs': success_probs, 'rank_error': error}
    return results


# computes the local likelihood of a given domain and the hierarchy
def local_log_likelihood(g, domain):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_node_probs(gp)
    likelihood = 0.0
    for n in gpp.nodes:
        likelihood += math.log(gpp.node[n]['reach_prob'])
    return likelihood

# computes the log likelihood of a hierarchy
def log_likelihood(g, domains):
    likelihood = 0.0
    for domain in domains:
        local_likelihood = local_log_likelihood(g, domain)
        likelihood += local_likelihood
    return likelihood















