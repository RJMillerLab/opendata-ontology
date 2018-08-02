import networkx as nx
import org.graph as orgg
import operator
import numpy as np
import math
from scipy import linalg


def compute_reachability_probs(gp, domains, tagdomains):
    selectionprobs = get_domains_selection_probs(tagdomains)
    top = list(nx.topological_sort(gp))
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = gp.copy()
    for domain in domains:
        g, tags = compute_tag_probs(h, domain, top)
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                success_probs[domain['name']] = selectionprobs[domain['tag']][domain['name']] * tag_dist[i][1]
        h = g.copy()
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


def get_partial_domain_edge_probs(g, domain, nodes, updates):
    gd = g.copy()
    sims = get_sims(gd, domain, updates)
    seen = []
    for p in nodes:
        for m in gd.predecessors(p):
            if m in seen:
                continue
            ts, sis = get_trans_prob_plus(gd, m, domain, sims)
            # or just update the trans prob of p
            for ch, prob in ts.items():
                gd[m][ch][domain['name']] = dict()
                gd[m][ch][domain['name']]['trans_prob_domain'] = prob
                gd[m][ch][domain['name']]['trans_sim_domain'] = sis[ch]
            seen.append(m)
    return gd


def get_sims(g, domain, nodes):
    sims = dict()
    for n in nodes:
        sims[n] = get_transition_sim(g.node[n]['rep'], domain['mean'])
    return sims


def get_domain_edge_probs(g, domain):
    gd = g.copy()
    leaves = orgg.get_leaves(gd)
    for p in gd.nodes:
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        if sum(ts.values()) > 1.0000001:
            print('improper: %f' % sum(ts.values()))
        for ch, prob in ts.items():
            gd[p][ch][domain['name']] = dict()
            gd[p][ch][domain['name']]['trans_prob_domain'] = prob
            gd[p][ch][domain['name']]['trans_sim_domain'] = sis[ch]
    return gd


def get_partial_domain_node_probs(g, domain, top, nodes):
    gd = g.copy()
    root = orgg.get_root(gd)
    to_use = list(nodes)
    for n in nodes:
        to_use.extend(list(g.predecessors(n)))
    to_use = list(set(to_use))
    for n in nodes:
        gd.node[n][domain['name']] = dict()
        if n == root:
            gd.node[n][domain['name']]['reach_prob_domain'] = 1.0
            gd.node[n][domain['name']]['reach_trans_domain'] = 1.0
        else:
            gd.node[n][domain['name']]['reach_prob_domain'] = 0.0
            gd.node[n][domain['name']]['reach_trans_domain'] = 0.0
    for p in top:
        if p not in to_use:
            continue
        for ch in list(gd.successors(p)):
            if ch not in nodes:
                continue
            gd.node[ch][domain['name']]['reach_prob_domain'] += gd.node[p][domain['name']]['reach_prob_domain']*                                       gd[p][ch][domain['name']]['trans_prob_domain']
            gd.node[ch][domain['name']]['reach_trans_domain'] += gd.node[p][domain['name']]['reach_trans_domain']*gd[p][ch][domain['name']]['trans_sim_domain']
            if gd.node[ch][domain['name']]['reach_prob_domain'] > 1.0:
                print('>0.1 %f' % gd.node[ch][domain['name']]['reach_prob_domain'])
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
    for p in top:
        for ch in list(gd.successors(p)):
            gd.node[ch][domain['name']]['reach_prob_domain'] += gd.node[p][domain['name']]['reach_prob_domain']*                                       gd[p][ch][domain['name']]['trans_prob_domain']
            gd.node[ch][domain['name']]['reach_trans_domain'] += gd.node[p][domain['name']]['reach_trans_domain']*gd[p][ch][domain['name']]['trans_sim_domain']
            if gd.node[ch][domain['name']]['reach_prob_domain'] > 1.0:
                print('>0.1 %f' % gd.node[ch][domain['name']]['reach_prob_domain'])
                gd.node[ch][domain['name']]['reach_prob_domain'] = 1.0
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


def recompute_success_prob(g, domains, nodes, updates, tagdomains):
    selectionprobs = get_domains_selection_probs(tagdomains)
    error = 0
    top = list(nx.topological_sort(g))
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = g.copy()
    for domain in domains:
        gp = get_partial_domain_edge_probs(h, domain, nodes, updates)
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
                success_probs[domain['name']] = selectionprobs[domain['tag']][domain['name']] * tag_dist[i][1]
        h = gpp.copy()
    for p in nodes:
        h.node[p]['reach_prob'] = 0.0
        for domain in domains:
            h.node[p]['reach_prob'] += h.node[p][domain['name']]['reach_prob_domain']
    for p in nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    error = sum(tag_ranks.values()) - len(domains)
    expected_success = sum(list(success_probs.values()))/float(len(domains))
    if expected_success == 0:
        print('zero expected_success.')
    print('hierarchy success prob: %f error: %d' % (expected_success, error))
    return expected_success, h, success_probs


def get_success_prob(g, domains, tagdomains):
    selectionprobs = get_domains_selection_probs(tagdomains)
    error = 0
    top = list(nx.topological_sort(g))
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = g.copy()
    for domain in domains:
        gp = get_domain_edge_probs(h, domain)
        gpp = get_domain_node_probs(gp, domain, top)
        tags = dict()
        for n in orgg.get_leaves(gpp):
            tags[gpp.node[n]['tag']] = gpp.node[n][domain['name']]['reach_prob_domain']

        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                success_probs[domain['name']] = selectionprobs[domain['tag']][domain['name']] * tag_dist[i][1]
                #print('tag prob: %f  selection prob: %f  %d' % (tag_dist[i][1], selectionprobs[domain['tag']][domain['name']], len(selectionprobs[domain['tag']])))
        h = gpp.copy()
    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0
        for domain in domains:
            h.node[p]['reach_prob'] += h.node[p][domain['name']]['reach_prob_domain']
    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))
    error = sum(tag_ranks.values()) - len(domains)
    expected_success = sum(list(success_probs.values()))/float(len(domains))
    if expected_success == 0:
        print('zero expected_success.')
    print('hierarchy success prob: %f error: %d' % (expected_success, error))
    return expected_success, h, success_probs


def evaluate(g, domains, tagdomains):
    error = 0
    h, tag_dists, tag_ranks, success_probs = compute_reachability_probs(g, domains, tagdomains)
    error = sum(tag_ranks.values()) - len(domains)
    expected_success = sum(list(success_probs.values()))/float(len(domains))
    print('hierarchy success prob: %f error: %d' % (expected_success, error))
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
    d2 = 0.0
    tps = dict()
    tps2 = dict()
    sis = dict()
    tsl = []
    ts = dict()
    sps = list(g.successors(p))
    for s in sps:
        m = get_transition_sim(g.node[s]['rep'], domain['mean'])
        tsl.append(m)
        ts[s] = m
    maxs = max(tsl)
    mins = min(tsl)
    for s in sps:
        if maxs == mins:
            tps[s] = math.exp(ts[s]-maxs)
        else:
            tps[s] = math.exp((ts[s]-mins)/(maxs-mins))
        tps2[s] = math.exp(5*ts[s])
        sis[s] = ts[s]
        d += tps[s]
        d2 += tps2[s]
    for s in sps:
        tps[s] = tps[s]/d
        tps2[s] = tps2[s]/d2
    return tps2, sis

def get_domains_selection_probs(tagdomains):
    probs = dict()
    for tag, domains in tagdomains.items():
        probs[tag] = dict()
        for target in domains:
            probs[tag][target['name']] = get_selection_probs(domains, target)
    return probs

def get_selection_probs(choices, domain):
    d, d2 = 0.0, 0.0
    tps, tps2, sis, ts = dict(), dict(), dict(), dict()
    tsl = []
    for s in choices:
        m = get_transition_sim(s['mean'], domain['mean'])
        tsl.append(m)
        ts[s['name']] = m
    maxs, mins = max(tsl), min(tsl)
    for s in choices:
        if maxs == mins:
            tps[s['name']] = math.exp(ts[s['name']]-maxs)
        else:
            tps[s['name']] = math.exp((ts[s['name']]-mins)/(maxs-mins))
        tps2[s['name']] = math.exp(5*ts[s['name']])
        sis[s['name']] = ts[s['name']]
        d += tps[s['name']]
        d2 += tps2[s['name']]
    return tps2[domain['name']]/d2




def get_trans_prob_plus(g, p, domain, sims):
    d = 0.0
    tps = dict()
    sis = dict()
    tsl = []
    ts = dict()
    sps = list(g.successors(p))
    for s in sps:
        if s in sims:
            m = sims[s]
        else:
            m = g[p][s][domain['name']]['trans_sim_domain']
        tsl.append(m)
        ts[s] = m
    #maxs = max(tsl)
    #mins = min(tsl)
    for s in sps:
        #if maxs == mins:
        #    tps[s] = math.exp(ts[s]-maxs)
        #else:
        #    tps[s] = math.exp((ts[s]-mins)/(maxs-mins))
        tps[s] = math.exp(5*ts[s])
        sis[s] = ts[s]
        d += tps[s]
    for s in sps:
        tps[s] = tps[s]/d
    return tps, sis


def get_improvement(init, final):
    imp = 0.0
    for t, p in init.items():
        imp += (final[t] - p)
    return imp






















