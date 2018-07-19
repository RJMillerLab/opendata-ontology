import networkx as nx
import org.graph as orgg
import operator
import numpy as np
import math
from scipy import linalg


def compute_reachability_probs(gp, domains):
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    for domain in domains:
        print('table: %s = %s' % (domain['name'], domain['tag']))
        tags = compute_tag_probs(gp, domain)
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                print('%s rank: %d' % (domain['tag'], i+1))
                success_probs[domain['name']] = tag_dist[i][1]
        print('-----------------------------')
    return tag_dists, tag_ranks, success_probs


def get_reachability_probs(gp, domains):
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    for domain in domains:
        tags = get_tag_probs(gp, domain)
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
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
            g.node[n]['population'] = list(node_vecs)
            g.node[n]['rep'] = np.mean(node_vecs, axis=0)
            g.node[n]['cov'] = np.cov(node_vecs)
            g.node[n]['mean'] = g.node[n]['rep']
            g.node[n]['det'] = linalg.det(g.node[n]['cov'])
        else:
            g.node[n]['rep'] = list(vecs[n])
            g.node[n]['population'] = [vecs[n]]
    return g


def compute_tag_probs(g, domain):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_domain_node_probs(gp)
    tags = dict()
    for n in orgg.get_leaves(gpp):
        tags[gpp.node[n]['tag']] = gpp.node[n]['reach_prob_domain']
    return tags


def get_tag_probs(g, domain):
    tags = dict()
    for n in orgg.get_leaves(g):
        tags[g.node[n]['tag']] = g.node[n][domain['name']]['reach_prob_domain']
    return tags


def get_domain_edge_probs(g, domain):
    gd = g.copy()
    # computing sims
    for e in gd.edges:
        p, ch = e[0], e[1]
        gd[p][ch]['trans_sim_domain'] = get_transition_sim(gd.node[ch]['rep'], domain['mean'])
    # computing softmax prob
    for e in gd.edges:
        p, ch = e[0], e[1]
        # softmax
        #d = float(sum([math.exp(gd[p][s]['trans_sim_domain']) for s in gd.successors(p)]))
        #gd[p][ch]['trans_prob_domain'] = math.exp(gd[p][ch]['trans_sim_domain'])/d
        d = float(sum([gd[p][s]['trans_sim_domain']+1.0 for s in gd.successors(p)]))
        gd[p][ch]['trans_prob_domain'] = (gd[p][ch]['trans_sim_domain']+1.0)/d
        #print('trans_sim p: %d ch: %d  %f  %f' % (p, ch, gd[p][ch]['trans_sim_domain'], gd[p][ch]['trans_prob_domain']))
    #print('------------------------')
    return gd


def get_domain_node_probs(g):
    gd = g.copy()
    for n in gd.node:
        if n == orgg.get_root(gd):
            gd.node[n]['reach_prob_domain'] = 1.0
            gd.node[n]['reach_trans_domain'] = 1.0
        else:
            gd.node[n]['reach_prob_domain'] = 0.0
            gd.node[n]['reach_trans_domain'] = 0.0
    top = list(nx.topological_sort(gd))
    leaves = orgg.get_leaves(gd)
    for p in top:
        if p in leaves:
            print('leaf node %d: %f  tag: %s' % (p, gd.node[p]['reach_prob_domain'], gd.node[p]['tag']))
        for ch in list(gd.successors(p)):
            gd.node[ch]['reach_prob_domain'] += gd.node[p]['reach_prob_domain']*gd[p][ch]['trans_prob_domain']
            gd.node[ch]['reach_trans_domain'] += gd.node[p]['reach_trans_domain']*gd[p][ch]['trans_sim_domain']
            print('p: %d ch: %d trans: %f prob: %f' % (p, ch, gd[p][ch]['trans_sim_domain'], gd.node[ch]['reach_prob_domain']))
            if gd.node[ch]['reach_prob_domain'] > 1.0:
                print('>0.1')
        print('-------------------')
    return gd


def get_transition_sim(vec1, vec2):
    # normalized cosine similarity
    c = cosine(vec1, vec2)
    return c


def cosine(vec1, vec2):
    dp = np.dot(vec1,vec2)
    c = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return c


def get_transition_prob_plus(vec1, mean, cov):
    #s = 0.0
    #for vec2 in vecs2:
    #    s += (1.0 - spatial.distance.cosine(vec1, vec2) + 1.0)/2.0
    #return s/float(len(vecs2))

    #m = 0.0
    #for vec2 in vecs2:
    #    d = (1.0 - spatial.distance.cosine(vec1, vec2) + 1.0)/2.0
    #    if d > m:
    #        m = d
    #return m
    return get_isa_sim(np.array(list(vec1)), mean, cov)

def get_isa_sim(vec, mean, cov, det):
    a = vec-mean
    c = np.exp(-0.5 * a * cov * np.transpose(a))
    d = 1.0/(((2.0*np.pi)**(len(vec)/2.0))*math.sqrt(det))
    f = d * c
    return f


def evaluate(g, domains):
    error = 0
    tag_dists, tag_ranks, success_probs = compute_reachability_probs(g, domains)
    print('tag ranks: {}'.format(tag_ranks))
    print('Computing reachability probs')
    error = sum(tag_ranks.values()) - len(domains)
    print('error: %d' % error)
    print('domain search success probs: ')
    print(success_probs)
    print('hierarchy success prob: %f' % (sum(list(success_probs.values()))/float(len(domains))))
    results = {'tag_dists': tag_dists, 'tag_ranks': tag_ranks, 'success_probs': success_probs, 'rank_error': error}
    return results


def get_state_probs(g, domains):
    h = g.copy()
    for domain in domains:
        gp = get_domain_edge_probs(h, domain)
        gpp = get_domain_node_probs(gp)
        for n in gpp.nodes:
            if 'reach_prob' not in gpp.node[n]:
                gpp.node[n]['reach_prob'] = gpp.node[n]['reach_prob_domain']
            else:
                gpp.node[n]['reach_prob'] += gpp.node[n]['reach_prob_domain']
        h = gpp
    state_probs = dict()
    for n in gpp.nodes:
        state_probs[n] = gpp.node[n]['reach_prob']
    return state_probs, h


# computes the local likelihood of a given domain and the hierarchy
def local_log_likelihood(g, domain):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_domain_node_probs(gp)
    likelihood = 0.0
    for n in gpp.nodes:
        likelihood += math.log(gpp.node[n]['reach_prob_domain'])
    return likelihood

# computes the log likelihood of a hierarchy
def log_likelihood(g, domains):
    likelihood = 0.0
    for domain in domains:
        local_likelihood = local_log_likelihood(g, domain)
        likelihood += local_likelihood
    return likelihood


























