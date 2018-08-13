import networkx as nx
import org.graph as orgg
import operator
import numpy as np
import math
from scipy import linalg
#import datetime
import multiprocessing
import random
from sklearn.cluster import KMeans
#from itertools import repeat


node_dom_sims = dict()
dom_selection_probs = dict()
h = nx.DiGraph()
leaves = []
top = []

def init(g, domains, tagdomains):
    global node_dom_sims, dom_selection_probs

    dom_selection_probs = get_domains_selection_probs(tagdomains)

    for n in g.nodes:
        node_dom_sims[n] = dict()
        for dom in domains:
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])


def update_node_dom_sims(g, domains, ns):
    for n in ns:
        for dom in domains:
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])


def compute_reachability_probs(gp, domains, tagdomains):
    top = list(nx.topological_sort(gp))
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = gp
    leaves = orgg.get_leaves(h)
    dsps = dict()
    for domain in domains:
        table = domain['name'][:domain['name'].rfind('_')]

        g, tags = compute_tag_probs(h, domain, top, leaves)
        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_dists[domain['tag']] = tag_dist
        for i in range(len(tag_dist)):
            if tag_dist[i][0]==domain['tag']:
                tag_ranks[domain['name']] = i + 1
                sp = dom_selection_probs[domain['tag']][domain['name']] * tag_dist[i][1]
                if table not in success_probs:
                    success_probs[table] = sp
                else:
                    success_probs[table] += sp
                dsps[domain['name']] = sp
        h = g.copy()
    print('dsp: %d  %f' % (len(dsps), sum(list(dsps.values()))/float(len(dsps))))
    print('success_prob: %d  %f' % (len(success_probs), sum(list(success_probs.values()))/float(len(success_probs))))
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


def compute_tag_probs(g, domain, top, leaves):
    gp = get_domain_edge_probs(g, domain, leaves)
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
    gd = g
    seen = []
    for p in nodes:
        for m in gd.predecessors(p):
            if m in seen:
                continue
            ts, sis = get_trans_prob(gd, m, domain)
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
        #sims[n] = get_transition_sim(g.node[n]['rep'], domain['mean'])
        sims[n] = node_dom_sims[n][domain['name']]
    return sims


def get_domain_edge_probs(g, domain, leaves):
    #gd = g.copy()
    gd = g
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
    gd = g
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
    gd = g
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


def recompute_local_success_prob(domain):

    global top, h, leaves, nodes, updates

    g = h.copy()

    reach_probs = dict()
    success_prob = 0.0

    gp = get_partial_domain_edge_probs(g, domain, nodes)
    gpp = get_partial_domain_node_probs(gp, domain, top, nodes)

    for n in leaves:
        if gpp.node[n]['tag'] == domain['tag']:
            success_prob = dom_selection_probs[domain['tag']][domain['name']] * gpp.node[n][domain['name']]['reach_prob_domain']

    for p in h.nodes:
        reach_probs[p] = h.node[p][domain['name']]['reach_prob_domain']

    return success_prob, reach_probs

def recompute_success_prob_likelihood(g, adomains, nodes, tagdomains, do, all_success_probs):
    domains = []
    if do:
        domains = get_domains_to_update(g, adomains, nodes, tagdomains)
        print('considering %d doms instead of %d.' % (len(domains), len(adomains)))
    else:
        domains = adomains

    top = list(nx.topological_sort(g))
    leaves = orgg.get_leaves(g)
    success_probs = dict()
    likelihood = 0.0
    h = g

    for p in nodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        table = domain['name'][:domain['name'].rfind('_')]

        gp = get_partial_domain_edge_probs(g, domain, nodes)
        gpp = get_partial_domain_node_probs(gp, domain, top, nodes)

        for n in leaves:
            if gpp.node[n]['tag'] == domain['tag']:
                sp = dom_selection_probs[domain['tag']][domain['name']] * gpp.node[n][domain['name']]['reach_prob_domain']
                if table not in success_probs:
                    success_probs[table] = sp
                else:
                    success_probs[table] += sp

        for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']
            likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])

        h = gpp

    for p in nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    # complete success_probs
    for d, p in all_success_probs.items():
        if d not in success_probs:
            success_probs[d] = p


    expected_success = sum(list(success_probs.values()))/float(len(success_probs))

    return expected_success, h.copy(), success_probs,  likelihood


def recompute_success_prob_likelihood_plus(g, domains, ns, ups):

    global top, h, leaves, nodes, updates
    top = list(nx.topological_sort(g))
    h = g
    leaves = orgg.get_leaves(g)
    nodes = ns
    updates = ups

    success_probs = []
    local_likelihoods = [0.0 for d in domains]
    pool = multiprocessing.Pool(5)

    results = pool.map(recompute_local_success_prob, domains)

    pool.close()
    pool.join()

    for p in nodes:
        h.node[p]['reach_prob'] = 0.0

    for r in results:
        success_probs.append(r[0])
        ll = 0.0
        for n, p in r[1].items():
            h.node[n]['reach_prob'] += p
            ll += math.log(p)
        local_likelihoods.append(ll)

    # computing reach probs
    for p in nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(success_probs)/float(len(domains))
    likelihood = sum(local_likelihoods)
    print('hierarchy success prob: %f and likelihood: %f' % (expected_success, likelihood))

    return expected_success, h.copy(), success_probs,  likelihood


def get_success_prob_likelihood(g, domains):

    top = list(nx.topological_sort(g))
    success_probs = dict()
    h = g
    leaves = orgg.get_leaves(g)
    likelihood = 0.0

    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        table = domain['name'][:domain['name'].rfind('_')]
        gp = get_domain_edge_probs(h, domain, leaves)
        gpp = get_domain_node_probs(gp, domain, top)

        for n in leaves:
            if gpp.node[n]['tag'] == domain['tag']:
                sp = dom_selection_probs[domain['tag']][domain['name']] * gpp.node[n][domain['name']]['reach_prob_domain']
                if table not in success_probs:
                    success_probs[table] = sp
                else:
                    success_probs[table] += sp

        for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']
            likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])

        h = gpp

    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    return expected_success, h, success_probs, likelihood


def get_success_prob_likelihood_plus(g, domains):

    global top, h, leaves
    top = list(nx.topological_sort(g))
    leaves = orgg.get_leaves(g)
    h = g

    success_probs = dict()
    local_likelihoods = [0.0 for d in domains]

    pool = multiprocessing.Pool(5)
    results = pool.map(get_local_success_prob, domains)
    pool.close()
    pool.join()

    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0

    for i in range(len(results)):
        r = results[i]
        success_probs[domains[i]['name']] = r[0]
        ll = 0.0
        for n, p in r[1].items():
            h.node[n]['reach_prob'] += p
            ll += math.log(p)
        local_likelihoods.append(ll)

    # computing reach probs
    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(domains))
    likelihood = sum(local_likelihoods)
    print('hierarchy success prob: %f and likelihood: %f' % (expected_success, likelihood))

    return expected_success, h.copy(), success_probs, likelihood


def get_local_success_prob(domain):

    global top, h, leaves

    reach_probs = dict()

    gp = get_domain_edge_probs(h, domain, leaves)
    gpp = get_domain_node_probs(gp, domain, top)

    for n in leaves:
        if gpp.node[n]['tag'] == domain['tag']:
            success_prob = dom_selection_probs[domain['tag']][domain['name']] * gpp.node[n][domain['name']]['reach_prob_domain']

    for p in gpp.nodes:
        reach_probs[p] = gpp.node[p][domain['name']]['reach_prob_domain']

    return success_prob, reach_probs


def evaluate(g, domains, tagdomains):
    error = 0
    h, tag_dists, tag_ranks, success_probs = compute_reachability_probs(g, domains, tagdomains)
    error = sum(tag_ranks.values()) - len(domains)
    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    print('hierarchy success prob: %f error: %d' % (expected_success, error))
    results = {'tag_dists': tag_dists, 'tag_ranks': tag_ranks, 'success_probs': success_probs, 'rank_error': error, 'expected_success': expected_success}
    return results


def get_state_probs(g, domains):
    #h = g.copy()
    h = g
    top = list(nx.topological_sort(g))
    leaves = orgg.get_leaves(g)
    for domain in domains:
        gp = get_domain_edge_probs(h, domain, leaves)
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
def local_log_likelihood(g, domain, top, leaves):
    gp = get_domain_edge_probs(g, domain, leaves)
    gpp = get_domain_node_probs(gp, domain, top)
    likelihood = 0.0
    for n in gpp.nodes:
        likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])
    #return gpp.copy(), likelihood
    return gpp, likelihood


# computes the log likelihood of a hierarchy
def log_likelihood(g, domains):
    top = list(nx.topological_sort(g))
    likelihood = 0.0
    #h = g.copy()
    h = g
    leaves = orgg.get_leaves(h)
    for domain in domains:
        gp, local_likelihood = local_log_likelihood(h, domain, top, leaves)
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
    branching_factor = len(sps)
    for s in sps:
        m = node_dom_sims[s][domain['name']]
        tsl.append(m)
        ts[s] = m
    maxs = max(tsl)
    mins = min(tsl)
    for s in sps:
        if maxs == mins:
            tps[s] = math.exp(ts[s]-maxs)
        else:
            tps[s] = math.exp((ts[s]-mins)/(maxs-mins))
        tps2[s] = math.exp((10.0/branching_factor)*ts[s]) # 5*
        sis[s] = ts[s]
        d += tps[s]
        d2 += tps2[s]
    for s in sps:
        tps[s] = tps[s]/d
        tps2[s] = tps2[s]/d2
    return tps2, sis

def get_domains_selection_probs(tagdomains):
    probs = dict()
    for tag, doms in tagdomains.items():
        probs[tag] = dict()
        for target in doms:
            probs[tag][target['name']] = get_selection_probs(doms, target)
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
    branching_factor = len(choices)
    for s in choices:
        if maxs == mins:
            tps[s['name']] = math.exp(ts[s['name']]-maxs)
        else:
            tps[s['name']] = math.exp((ts[s['name']]-mins)/(maxs-mins))
        tps2[s['name']] = math.exp((10.0/branching_factor)*ts[s['name']]) # 5*
        sis[s['name']] = ts[s['name']]
        d += tps[s['name']]
        d2 += tps2[s['name']]
    return tps2[domain['name']]/d2


def get_improvement(init, final):
    imp = 0.0
    for t, p in init.items():
        imp += (final[t] - p)
    return imp


def get_domains_to_update(g, domains, nodes, tagdomains):
    updomains = []
    dnames = []
    leaves = orgg.get_leaves(g)
    leaf_nodes = set()
    for n in nodes:
        ds = set(nx.descendants(g, n)).intersection(set(leaves))
        if n in leaves:
            ds = ds.union({n})
        leaf_nodes = leaf_nodes.union(ds)
    for s in leaf_nodes:
        for d in tagdomains[g.node[s]['tag']]:
            if d['name'] not in dnames:
                updomains.append(d)
                dnames.append(d['name'])
    return updomains


def get_dimensions(tags, vecs, n_dims):
    kmeans = KMeans(n_clusters=n_dims, random_state=random.randint(1,1000)).fit(vecs)
    dims = dict()
    for i in range(len(kmeans.labels_)):
        c = kmeans.labels_[i]
        if c not in dims:
            dims[c] = []
        dims[c].append(tags[i])
    return dims


















