import networkx as nx
import org.graph as orgg
import operator
import numpy as np
import math
from scipy import linalg


def compute_reachability_probs(gp, domains):
    top = list(nx.topological_sort(gp))
    changed = 0
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = gp.copy()
    for domain in domains:
        #print('table: %s = %s' % (domain['name'], domain['tag']))
        g, tags = compute_tag_probs(h, domain, top)
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                #print('%s rank: %d' % (domain['tag'], i+1))
                success_probs[domain['name']] = tag_dist[i][1]
        #print('-----------------------------')
        h = g.copy()
    ols = dict()
    for p in h.nodes:
        if 'reach_prob' not in h.node[p]:
            ols[p] = 0.0
        else:
            ols[p] = h.node[p]['reach_prob']
        h.node[p]['reach_prob'] = 0.0
        for domain in domains:
            h.node[p]['reach_prob'] += h.node[p][domain['name']]['reach_prob_domain']
    ls = orgg.get_leaves(h)
    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))
        l = ""
        if p in ls:
            l = 'leaf'
        if h.node[p]['reach_prob'] > ols[p]:
            changed += 1
            print('[%d] old p: %f  new p: %f.   UP - %s' % (p, ols[p], h.node[p]['reach_prob'], l))
        elif h.node[p]['reach_prob'] < ols[p]:
            changed += 1
            print('[%d] old p: %f  new p: %f    DOWN - %s' % (p, ols[p], h.node[p]['reach_prob'], l))
        else:
            print('[%d] old p: %f  new p: %f - %s' % (p, ols[p], h.node[p]['reach_prob'], l))
    print('prob changed for %d nodes.' % changed)
    return h, tag_dists, tag_ranks, success_probs


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


def compute_tag_probs(g, domain, top):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_domain_node_probs(gp, domain, top)
    tags = dict()
    for n in orgg.get_leaves(gpp):
        tags[gpp.node[n]['tag']] = gpp.node[n][domain['name']]['reach_prob_domain']
    return gpp, tags


def get_tag_probs(g, domain):
    tags = dict()
    for n in orgg.get_leaves(g):
        tags[g.node[n]['tag']] = g.node[n][domain['name']]['reach_prob_domain']
    return tags


def get_partial_domain_edge_probs(g, domain, nodes):
    gd = g.copy()
    leaves = orgg.get_leaves(gd)
    for p in nodes:
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            gd[p][ch][domain['name']] = dict()
            gd[p][ch][domain['name']]['trans_prob_domain'] = prob
            gd[p][ch][domain['name']]['trans_sim_domain'] = sis[ch]
    return gd




def get_domain_edge_probs(g, domain):
    gd = g.copy()
    leaves = orgg.get_leaves(gd)
    for p in gd.nodes:
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            gd[p][ch][domain['name']] = dict()
            gd[p][ch][domain['name']]['trans_prob_domain'] = prob
            gd[p][ch][domain['name']]['trans_sim_domain'] = sis[ch]
    return gd


def get_partial_domain_node_probs(g, domain, top, nodes):
    gd = g.copy()
    root = orgg.get_root(gd)
    for n in nodes:
        gd.node[n][domain['name']] = dict()
        if n == root:
            gd.node[n][domain['name']]['reach_prob_domain'] = 1.0
            gd.node[n][domain['name']]['reach_trans_domain'] = 1.0
        else:
            gd.node[n][domain['name']]['reach_prob_domain'] = 0.0
            gd.node[n][domain['name']]['reach_trans_domain'] = 0.0
    for p in top:
        if p not in nodes:
            continue
        for ch in list(gd.successors(p)):
            gd.node[ch][domain['name']]['reach_prob_domain'] += gd.node[p][domain['name']]['reach_prob_domain']*                                       gd[p][ch][domain['name']]['trans_prob_domain']
            gd.node[ch][domain['name']]['reach_trans_domain'] += gd.node[p][domain['name']]['reach_trans_domain']*gd[p][ch][domain['name']]['trans_sim_domain']
            if gd.node[ch][domain['name']]['reach_prob_domain'] > 1.0:
                print('>0.1')
                gd.node[ch][domain['name']]['reach_prob_domain'] = 1.0
    return gd


def get_domain_node_probs(g, domain, top):
    gd = g.copy()
    root = orgg.get_root(gd)
    for n in gd.node:
        gd.node[n][domain['name']] = dict()
        if n == root:
            gd.node[n][domain['name']]['reach_prob_domain'] = 1.0
            gd.node[n][domain['name']]['reach_trans_domain'] = 1.0
        else:
            gd.node[n][domain['name']]['reach_prob_domain'] = 0.0
            gd.node[n][domain['name']]['reach_trans_domain'] = 0.0
    #top = list(nx.topological_sort(gd))
    #leaves = orgg.get_leaves(gd)
    for p in top:
        #if p in leaves:
        #    print('leaf node %d: %f  tag: %s' % (p, gd.node[p]['reach_prob_domain'], gd.node[p]['tag']))
        for ch in list(gd.successors(p)):
            gd.node[ch][domain['name']]['reach_prob_domain'] += gd.node[p][domain['name']]['reach_prob_domain']*                                       gd[p][ch][domain['name']]['trans_prob_domain']
            gd.node[ch][domain['name']]['reach_trans_domain'] += gd.node[p][domain['name']]['reach_trans_domain']*gd[p][ch][domain['name']]['trans_sim_domain']
            #print('p: %d ch: %d trans sim: %f trans prob: %f reach prob: %f' % (p, ch, gd[p][ch][domain['name']]['trans_sim_domain'], gd[p][ch][domain['name']]['trans_prob_domain'], gd.node[ch][domain['name']]['reach_prob_domain']))
            if gd.node[ch][domain['name']]['reach_prob_domain'] > 1.0:
                print('>0.1')
                gd.node[ch][domain['name']]['reach_prob_domain'] = 1.0
        #print('-------------------')
    return gd


def get_transition_sim(vec1, vec2):
    # cosine similarity
    c = max(0.000001, cosine(vec1, vec2))
    return c


def cosine(vec1, vec2):
    dp = np.dot(vec1,vec2)
    c = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return c


def get_transition_sim_plus(vecs2, vec1):
    m = 0.0
    for vec2 in vecs2:
        d = max(0.000001, cosine(vec1, vec2))
        if d > m:
            m = d
    return m


def get_isa_sim(vec, mean, cov, det):
    a = vec-mean
    c = np.exp(-0.5 * a * cov * np.transpose(a))
    d = 1.0/(((2.0*np.pi)**(len(vec)/2.0))*math.sqrt(det))
    f = d * c
    return f


def recompute_success_prob_width(g, domains, nodes):
    error = 0
    changed = 0
    top = list(nx.topological_sort(g))
    print('Computing success probs')
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = g.copy()
    for domain in domains:
        gp = get_partial_domain_edge_probs(h, domain, nodes)
        gpp = get_partial_domain_node_probs(gp, domain, top, nodes)
        tags = dict()
        for n in orgg.get_leaves(gpp):
            tags[gpp.node[n]['tag']] = gpp.node[n][domain['name']]['reach_prob_domain']
        ##
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                success_probs[domain['name']] = tag_dist[i][1]
        h = gpp.copy()
    ols = dict()
    for p in nodes:
        if 'reach_prob' not in h.node[p]:
            ols[p] = 0.0
        else:
            ols[p] = h.node[p]['reach_prob']
        h.node[p]['reach_prob'] = 0.0
        for domain in domains:
            h.node[p]['reach_prob'] += h.node[p][domain['name']]['reach_prob_domain']
    ls = orgg.get_leaves(h)
    for p in nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))
        l = ""
        if p in ls:
            l = 'leaf'
        if h.node[p]['reach_prob'] > ols[p]:
            changed += 1
            #print('[%d] old p: %f  new p: %f.   UP - %s' % (p, ols[p], h.node[p]['reach_prob'], l))
        elif h.node[p]['reach_prob'] < ols[p]:
            changed += 1
            #print('[%d] old p: %f  new p: %f    DOWN - %s' % (p, ols[p], h.node[p]['reach_prob'], l))
        else:
            print('[%d] old p: %f  new p: %f - %s' % (p, ols[p], h.node[p]['reach_prob'], l))
    print('prob changed for %d nodes.' % changed)
    ##
    error = sum(tag_ranks.values()) - len(domains)
    print('error: %d' % error)
    expected_success = sum(list(success_probs.values()))/float(len(domains))
    print('hierarchy success prob: %f' % (expected_success))
    return expected_success, h




def success_prob(g, domains):
    error = 0
    print('Computing success probs')
    h, tag_dists, tag_ranks, success_probs = compute_reachability_probs(g, domains)
    error = sum(tag_ranks.values()) - len(domains)
    print('error: %d' % error)
    expected_success = sum(list(success_probs.values()))/float(len(domains))
    print('hierarchy success prob: %f' % (expected_success))
    return expected_success, h


def evaluate(g, domains):
    error = 0
    h, tag_dists, tag_ranks, success_probs = compute_reachability_probs(g, domains)
    #print('tag ranks: {}'.format(tag_ranks))
    print('Computing reachability probs')
    error = sum(tag_ranks.values()) - len(domains)
    print('error: %d' % error)
    #print('domain search success probs: ')
    #print(success_probs)
    expected_success = sum(list(success_probs.values()))/float(len(domains))
    print('hierarchy success prob: %f' % (expected_success))
    results = {'tag_dists': tag_dists, 'tag_ranks': tag_ranks, 'success_probs': success_probs, 'rank_error': error, 'expected_success': expected_success}
    return results


def get_state_probs(g, domains):
    h = g.copy()
    top = list(nx.topological_sort(g))
    for domain in domains:
        gp = get_domain_edge_probs(h, domain)
        gpp = get_domain_node_probs(gp, domain, top)
        for n in gpp.nodes:
            if 'reach_prob' not in gpp.node[n]:
                gpp.node[n]['reach_prob'] = gpp.node[n][domain['name']]['reach_prob_domain']
            else:
                gpp.node[n]['reach_prob'] += gpp.node[n][domain['name']]['reach_prob_domain']
        h = gpp
    state_probs = dict()
    for n in gpp.nodes:
        gpp.node[n]['reach_prob'] = gpp.node[n]['reach_prob'] / float(len(domains))
        state_probs[n] = gpp.node[n]['reach_prob']/float(len(domains))
    return h


# computes the local likelihood of a given domain and the hierarchy
def local_log_likelihood(g, domain, top):
    gp = get_domain_edge_probs(g, domain)
    gpp = get_domain_node_probs(gp, domain, top)
    likelihood = 0.0
    for n in gpp.nodes:
        likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])
    return gpp.copy(), likelihood

# computes the log likelihood of a hierarchy
def log_likelihood(g, domains):
    top = list(nx.topological_sort(g))
    likelihood = 0.0
    h = g.copy()
    for domain in domains:
        gp, local_likelihood = local_log_likelihood(h, domain, top)
        likelihood += local_likelihood
        h = gp
    ols = dict()
    for p in h.nodes:
        if 'reach_prob' not in h.node[p]:
            ols[p] = 0.0
        else:
            ols[p] = h.node[p]['reach_prob']
        h.node[p]['reach_prob'] = 0.0
        for domain in domains:
            h.node[p]['reach_prob'] += h.node[p][domain['name']]['reach_prob_domain']
    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))
        print('[%d] old ll: %f  new ll: %f.' % (p, ols[p], h.node[p]['reach_prob']))

    return likelihood, h


def get_trans_prob(g, p, domain):
    d = 0.0
    tps = dict()
    sis = dict()
    tsl = []
    ts = dict()
    sps = list(g.successors(p))
    for s in sps:
        m = get_transition_sim(g.node[s]['rep'], domain['mean'])
        #m = get_transition_sim_plus(g.node[s]['population'], domain['mean'])
        tsl.append(m)
        ts[s] = m
    maxs = max(tsl)
    mins = min(tsl)
    for s in sps:
        if maxs == mins:
            tps[s] = math.exp(ts[s]-maxs)
        else:
            tps[s] = math.exp((ts[s]-mins)/(maxs-mins))
        #tps[s] = (ts[s]-mins)/(maxs-mins)
        sis[s] = ts[s]
        d += tps[s]
    for s in sps:
        tps[s] = tps[s]/d
    return tps, sis

























