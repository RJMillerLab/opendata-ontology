import networkx as nx
import org.graph as orgg
import operator
import numpy as np
import math
#import datetime
import multiprocessing
import random
from sklearn.cluster import KMeans
#from itertools import repeat
import json
from sklearn.metrics import silhouette_score

node_dom_sims = dict()
dom_selection_probs = dict()
h = nx.DiGraph()
leaves = []
top = []
gamma = 10.0

def init(g, domains, tagdomains, simfile, tgparam=10.0):
    #print('trans sim > 0.5')
    #print('reach prob > 0.1')
    print('sanity check of domain prunning')
    global node_dom_sims, dom_selection_probs, dom_sims, gamma
    gamma = float(tgparam)
    print('gamma: %f' % gamma)

    dom_sims = json.load(open(simfile, 'r'))

    #dom_selection_probs = get_domains_selection_probs(tagdomains)

    for n in g.nodes:
        node_dom_sims[n] = dict()
        for dom in domains:
            #node_dom_sims[n][dom['name']] = np.random.random_sample()
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])

    print('done init')


def update_node_dom_sims(g, domains, ns):
    for n in ns:
        for dom in domains:
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])


def compute_reachability_probs_plus(gp, domains, tagdomains, domainclouds, dtype, domaintags):
    top = list(nx.topological_sort(gp))
    success_probs = dict()
    success_probs_intersect = dict()
    h = gp
    leaves = orgg.get_leaves(h)
    dom_target_sims = []
    reachable_dom_probs = []
    results = dict()
    # building a domain index on domain names
    domain_index = dict()
    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = []
        domain_index[domains[i]['name']].append(i)
    # evaluation
    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0
    samedom = 0
    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        for c in list(domainclouds[domain['name']].keys()):
            if c not in domain_index:
                continue
            for di in domain_index[c]:
                accepted_tags.extend(domaintags[domains[di]['name']])
                #accepted_tags.append(domains[di]['tag'])
        accepted_tags = list(set(accepted_tags))

        g, tags = compute_tag_probs(h, domain, top, leaves)

        tag_dist = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        # finding the most reachable domain
        max_reached_dom_prob = 0.0
        max_reached_dom_sim = 0.0
        most_reachable_dom = ''
        for i in range(len(tag_dist)):
            if tag_dist[i][0] not in accepted_tags:
                continue
            selps, sims = get_dom_trans_prob(tagdomains[tag_dist[i][0]], domain)
            #selps = dom_target_sims[tag_dist[i][0]]][domain['name']]
            for d in tagdomains[tag_dist[i][0]]:
                if d['name'] not in domainclouds[domain['name']]:
                    continue
                sp = selps[d['name']] * tag_dist[i][1]
                if sp > max_reached_dom_prob:
                    max_reached_dom_prob = sp
                    most_reachable_dom = d['name']
        reachable_dom_probs.append(max_reached_dom_prob)
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        sp = max_reached_dom_prob
        if table not in success_probs:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp

        if domain['name'] == most_reachable_dom:
            samedom += 1

        results[domain['name']] = {'most_reachable_dom': most_reachable_dom, 'dom_target_sim': max_reached_dom_sim, 'reachable_dom_prob': max_reached_dom_prob}

        h = g.copy()

        #t1 = datetime.datetime.now()
        #se2 += ((t1-t0).microseconds/1000)

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1-success_probs_intersect[t]
        #if success_probs[t] != success_probs_intersect[t]:
            #success_probs[t] -= success_probs_intersect[t]
        if success_probs[t] > 1.0:
             print('table %s has sp > 1.0.' % t)
             success_probs[t] = 1.0

    return dom_target_sims, reachable_dom_probs, success_probs


def compute_reachability_probs(gp, domains, tagdomains, dtype):
    top = list(nx.topological_sort(gp))
    tag_ranks = dict()
    tag_dists = dict()
    success_probs = dict()
    h = gp
    leaves = orgg.get_leaves(h)
    dsps = dict()
    # evaluation
    for domain in domains:
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
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
                if success_probs[table] > 1.0:
                    success_probs[table] = 1.0

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


def add_node_vecs(g, vecs, tags):
    leaves = orgg.get_leaves(g)
    for n in g.nodes:
        if n not in leaves:
            node_vecs = vecs[np.array(list(leaves.intersection(nx.descendants(g,n))))]
            node_tags = np.array(tags)[np.array(list(leaves.intersection(nx.descendants(g,n))))]
            g.node[n]['population'] = list(node_vecs)
            g.node[n]['tags'] = list(node_tags)
            g.node[n]['rep'] = np.mean(node_vecs, axis=0)
            #g.node[n]['cov'] = np.cov(node_vecs)
            g.node[n]['mean'] = g.node[n]['rep']
            #g.node[n]['det'] = linalg.det(g.node[n]['cov'])
        else:
            g.node[n]['rep'] = list(vecs[n])
            g.node[n]['population'] = [vecs[n]]
            g.node[n]['tags'] = [tags[n]]
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
        #if sum(ts.values()) > 1.0000001:
        #    print('improper: %f' % sum(ts.values()))
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


def recompute_success_prob_likelihood_fuzzy(g, adomains, nodes, tagdomains, do, all_success_probs, domainclouds, dtype):
    domains = []
    if do:
        domains, dnames = get_domains_to_update(g, adomains, nodes, tagdomains)
        print('considering %d doms instead of %d.' % (len(domains), len(adomains)))
    else:
        domains = list(adomains)

    top = list(nx.topological_sort(g))
    leaves = orgg.get_leaves(g)
    success_probs = dict()
    success_probs_intersect = dict()
    likelihood = 0.0
    dom_target_sims = []
    reachable_dom_probs = []
    #find_target_probs = []
    samedom = 0
    # building a domain index on domain names
    domain_index = dict()
    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = []
        domain_index[domains[i]['name']].append(i)

    h = g

    for p in nodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        for c in list(domainclouds[domain['name']].keys()):
            for di in domain_index[c]:
                accepted_tags.append(domains[di]['tag'])

        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        gp = get_partial_domain_edge_probs(g, domain, nodes)
        gpp = get_partial_domain_node_probs(gp, domain, top, nodes)

        # finding the most reachable domain
        max_reached_dom_prob = 0.0
        max_reached_dom_sim = 0.0
        most_reachable_dom = ''

        for n in leaves:
            if gpp.node[n]['tag'] not in accepted_tags:
                continue
            #if gpp.node[n]['tag'] == domain['tag']:
                #find_target_probs.append(dom_selection_probs[domain['tag']][domain['name']] * gpp.node[n][domain['name']]['reach_prob_domain'])
            selps, sims = get_dom_trans_prob(tagdomains[gpp.node[n]['tag']], domain)
            for d in tagdomains[gpp.node[n]['tag']]:
                if d['name'] not in domainclouds[domain['name']]:
                    continue
                sp = selps[d['name']] * gpp.node[n][domain['name']]['reach_prob_domain']
                if sp > max_reached_dom_prob:
                    max_reached_dom_prob = sp
                    max_reached_dom_sim = sims[d['name']]
                    most_reachable_dom = d['name']
        dom_target_sims.append(max_reached_dom_sim)
        reachable_dom_probs.append(max_reached_dom_prob)
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        sp = max_reached_dom_prob

        if table not in success_probs:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp

        if domain['name'] == most_reachable_dom:
            samedom += 1


        for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']
            likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])

        h = gpp

    for p in nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        # don't use the union if a table has only one table
        success_probs[t] = 1-success_probs_intersect[t]
        #if success_probs[t] != success_probs_intersect[t]:
            #success_probs[t] -= success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0

    # complete success_probs
    for d, p in all_success_probs.items():
        if d not in success_probs:
            success_probs[d] = p
        if p != success_probs[d]:
            print('inconsistency in success probs old %f new %f' % (p, success_probs[d]))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))

    return expected_success, h.copy(), success_probs,  likelihood




def recompute_success_prob_likelihood(g, adomains, nodes, tagdomains, do, all_success_probs, dtype):
    domains = []
    if do:
        domains, dnames = get_domains_to_update(g, adomains, nodes, tagdomains)
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
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
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
                if success_probs[table] > 1.0:
                    print('table %s has sp > 1.0.' % table)
                    success_probs[table] = 1.0

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



def get_success_prob_likelihood(g, domains, dtype):

    top = list(nx.topological_sort(g))
    success_probs = dict()
    h = g
    leaves = orgg.get_leaves(g)
    likelihood = 0.0

    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
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
                if success_probs[table] > 1.0:
                    success_probs[table] = 1.0

        for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']
            likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])

        h = gpp

    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))

    return expected_success, h, success_probs, likelihood


def get_success_prob_likelihood_partial_plus(g, adomains, tagdomains, domainclouds, dtype, domaintags, nodes, update_head,  prev_success_probs, prev_domain_success_probs):

    active_domains, dnames = get_domains_to_update(g.copy(), adomains, nodes, tagdomains, domainclouds, update_head, domaintags)
    print('exact doms: %d vs. %d' % (len(active_domains), len(adomains)))

    expected_success1, h1, success_probs1, likelihood1, domain_success_probs1 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, dnames)

    return expected_success1, h1.copy(), success_probs1, likelihood1, domain_success_probs1



def get_success_prob_likelihood_partial(g, adomains, tagdomains, domainclouds, dtype, domaintags, nodes, update_head,  prev_success_probs, prev_domain_success_probs):

    active_domains, dnames = get_domains_to_update(g.copy(), adomains, nodes, tagdomains, domainclouds, update_head, domaintags)
    print('exact doms: %d vs. %d' % (len(active_domains), len(adomains)))

    expected_success1, h1, success_probs1, likelihood1, domain_success_probs1 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, dnames)

    #expected_success3, h3, success_probs3, likelihood3, domain_success_probs3 = get_success_prob_likelihood_fuzzy(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags)
    expected_success3, h3, success_probs3, likelihood3, domain_success_probs3 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype,      domaintags, prev_success_probs, prev_domain_success_probs, adomains, [dom['name'] for dom in adomains])

    if expected_success3 != expected_success1:
        print('mistake in computing success prob: exact: %f and prune: %f' % (expected_success3, expected_success1))
        ws = 0
        for d, p in domain_success_probs3.items():
            if p!=domain_success_probs1[d]:
                if d in dnames:
                    print('wrong comp of sp dom')
                else:
                    for n in h1.nodes:
                        if h1.node[n][d]['reach_prob_domain'] != h3.node[n][d]['reach_prob_domain']:
                            if n not in nodes:# list(nx.descendants(h1, update_head)):
                                print('wrong potential nodes %d' % n)
                            else:
                                print('n in nodes and still wrong')
                    ws += 1
        print('%d wrong active dom calc out of %d' % (ws, len(adomains)))

    return expected_success1, h1.copy(), success_probs1, likelihood1, domain_success_probs1


def get_success_prob_prune_domains(g, adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, update_domain_names):

    domains = list(active_domains)

    h = g
    top = list(nx.topological_sort(g))

    success_probs = dict()
    success_probs_intersect = dict()
    domain_success_probs = dict()
    likelihood = 0.0

    # reset states' reachability
    for p in h.nodes:
        h.node[p]['reach_prob'] = 0
    # the reachability prob of domains that are not candidates for update remains intact
    for dom in adomains:
        if dom['name'] in update_domain_names:
            continue

        table = ''
        if dtype == 'synthetic':
            colid = int(dom['name'][dom['name'].rfind('_')+1:])
            table = dom['name'][:dom['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = dom['name'][:dom['name'].rfind('_')]

        sp = prev_domain_success_probs[dom['name']]
        if table not in success_probs_intersect:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp
        domain_success_probs[dom['name']] = prev_domain_success_probs[dom['name']]
        # accumulate the reachability of non-updatable domains
        for p in h.nodes:
            h.node[p]['reach_prob'] += h.node[p][dom['name']]['reach_prob_domain']
            likelihood += math.log(h.node[p][dom['name']]['reach_prob_domain'])

    leaves = orgg.get_leaves(g)
    samedom = 0
    dom_target_sims = []
    reachable_dom_probs = []
    # building a domain index on domain names
    domain_index = dict()
    for i in range(len(adomains)):
        if adomains[i]['name'] not in domain_index:
            domain_index[adomains[i]['name']] = []
        domain_index[adomains[i]['name']].append(i)
    #
    inc_count = 0
    dec_count = 0
    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        for c in list(domainclouds[domain['name']].keys()):
            if c not in domain_index:
                print('domain cloud not in index.')
                continue
            for di in domain_index[c]:
                accepted_tags.extend(domaintags[adomains[di]['name']])
        accepted_tags = list(set(accepted_tags))
        #
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        gp = get_domain_edge_probs(h, domain, leaves)
        gpp = get_domain_node_probs(gp, domain, top)
        # finding the most reachable domain
        max_reached_dom_prob = 0.0
        max_reached_dom_sim = 0.0
        most_reachable_dom = ''
        #
        for n in leaves:
            if gpp.node[n]['tag'] not in accepted_tags:
                continue
            if domain['name'] not in list(domainclouds[domain['name']].keys()):
                print('no %d' % len(list(domainclouds[domain['name']].keys())))
            selps, sims = get_dom_trans_prob(tagdomains[gpp.node[n]['tag']], domain)
            for d in tagdomains[gpp.node[n]['tag']]:
                if d['name'] not in domainclouds[domain['name']]:
                    continue
                sp = selps[d['name']] * gpp.node[n][domain['name']]['reach_prob_domain']
                if sp > max_reached_dom_prob:
                    max_reached_dom_prob = sp
                    max_reached_dom_sim = sims[d['name']]
                    most_reachable_dom = d['name']
        dom_target_sims.append(max_reached_dom_sim)
        reachable_dom_probs.append(max_reached_dom_prob)
        sp = max_reached_dom_prob

        if domain['name'] in domain_success_probs:
            print('rep dom alert')
        domain_success_probs[domain['name']] = sp
        if domain_success_probs[domain['name']] > prev_domain_success_probs[domain['name']]:
            inc_count += 1
        else:
            dec_count += 1

        if table not in success_probs:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp

        if domain['name'] == most_reachable_dom:
            samedom += 1

        for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']
            likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])

        h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0


    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(adomains))

    if len(success_probs) != len(prev_success_probs) or len(prev_domain_success_probs) != len(domain_success_probs):
        print('have not looked at all doms and tables.')
        print('len(success_probs): %d  len(prev_success_probs): %d len(prev_domain_success_probs): %d len(domain_success_probs: %d' % (len(success_probs), len(prev_success_probs), len(prev_domain_success_probs), len(domain_success_probs)))
    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    print('inc: %d dec: %d out of %d (%d)' % (inc_count, dec_count, len(domains), len(adomains)))

    return expected_success, h, success_probs, likelihood, domain_success_probs





def get_success_prob_likelihood_fuzzy(g, domains, tagdomains, domainclouds, dtype, domaintags):

    top = list(nx.topological_sort(g))
    success_probs = dict()
    domain_success_probs = dict()
    success_probs_intersect = dict()
    h = g
    leaves = orgg.get_leaves(g)
    likelihood = 0.0
    samedom = 0
    dom_target_sims = []
    reachable_dom_probs = []
    # building a domain index on domain names
    domain_index = dict()
    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = []
        domain_index[domains[i]['name']].append(i)
    #
    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        for c in list(domainclouds[domain['name']].keys()):
            if c not in domain_index:
                continue
            for di in domain_index[c]:
                #accepted_tags.append(domains[di]['tag'])
                accepted_tags.extend(domaintags[domains[di]['name']])
        accepted_tags = list(set(accepted_tags))
        #
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        gp = get_domain_edge_probs(h, domain, leaves)
        gpp = get_domain_node_probs(gp, domain, top)
        # finding the most reachable domain
        max_reached_dom_prob = 0.0
        max_reached_dom_sim = 0.0
        most_reachable_dom = ''
        #
        for n in leaves:
            if gpp.node[n]['tag'] not in accepted_tags:
                continue
            selps, sims = get_dom_trans_prob(tagdomains[gpp.node[n]['tag']], domain)
            for d in tagdomains[gpp.node[n]['tag']]:
                if d['name'] not in domainclouds[domain['name']]:
                    continue
                sp = selps[d['name']] * gpp.node[n][domain['name']]['reach_prob_domain']
                if sp > max_reached_dom_prob:
                    max_reached_dom_prob = sp
                    max_reached_dom_sim = sims[d['name']]
                    most_reachable_dom = d['name']
        dom_target_sims.append(max_reached_dom_sim)
        reachable_dom_probs.append(max_reached_dom_prob)
        #if max_reached_dom_prob < find_target_probs[-1]:
        #    print('weird %f < %f' % (max_reached_dom_prob, find_target_probs[-1]))
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        sp = max_reached_dom_prob

        domain_success_probs[domain['name']] = sp


        if table not in success_probs:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp

        if domain['name'] == most_reachable_dom:
            samedom += 1

        for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']
            likelihood += math.log(gpp.node[n][domain['name']]['reach_prob_domain'])

        h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1-success_probs_intersect[t]
        #if success_probs[t] != success_probs_intersect[t]:
            #success_probs[t] -= success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0


    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    return expected_success, h, success_probs, likelihood, domain_success_probs


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



def fuzzy_evaluate(g, domains, tagdomains, domainclouds, dtype, domaintags):
    print('fuzzy_evaluate')
    dom_reach_sims, dom_reach_probs, success_probs = compute_reachability_probs_plus(g, domains, tagdomains, domainclouds, dtype, domaintags)
    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    print('hierarchy success prob fuzzy: %f' % expected_success)
    results = {'success_probs': success_probs, 'dom_reach_sims': dom_reach_sims, 'dom_reach_probs': dom_reach_probs}
    return results



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


def get_dom_trans_prob(choices, domain):
    global gamma
    d2 = 0.0
    tps2 = dict()
    sis = dict()
    tsl = []
    ts = dict()
    branching_factor = len(choices)
    for s in choices:
        #m = get_transition_sim(s['mean'], domain['mean'])
        m = 0.0
        if s['name'] in dom_sims:
            if domain['name'] in dom_sims[s['name']]:
                m = dom_sims[s['name']][domain['name']]
        tsl.append(m)
        ts[s['name']] = m
    for s in choices:
        #tps2[s['name']] = math.exp(5.0*ts[s['name']])
        tps2[s['name']]  = math.exp((gamma/branching_factor)*ts[s['name']])
        sis[s['name']] = ts[s['name']]
        d2 += tps2[s['name']]
    for s in choices:
        tps2[s['name']] = (tps2[s['name']]/d2)
    return tps2, sis




def get_trans_prob(g, p, domain):
    global gamma
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
    #maxs = max(tsl)
    #mins = min(tsl)
    for s in sps:
        #if maxs == mins:
        #    tps[s] = math.exp(ts[s]-maxs)
        #else:
        #    tps[s] = math.exp((ts[s]-mins)/(maxs-mins))
        tps2[s] = math.exp(5.0*ts[s])#(10.0/branching_factor)*ts[s])
        tps[s]  = math.exp((gamma/branching_factor)*ts[s])
        sis[s] = ts[s]
        d += tps[s]
        d2 += tps2[s]
    for s in sps:
        tps[s] = tps[s]/d
        tps2[s] = tps2[s]/d2

    return tps, sis

def get_domains_selection_probs(tagdomains):
    probs = dict()
    for tag, doms in tagdomains.items():
        probs[tag] = dict()
        for target in doms:
            probs[tag][target['name']] = get_selection_probs(doms, target)
    return probs


def get_selection_probs(choices, domain):
    global gamma
    d = 0.0
    tps, sis, ts = dict(), dict(), dict()
    tsl = []
    for s in choices:
        #m = get_transition_sim(s['mean'], domain['mean'])
        m = 0.0
        if s['name'] in dom_sims:
            if domain['name'] in dom_sims[s['name']]:
                m = dom_sims[s['name']][domain['name']]
        tsl.append(m)
        ts[s['name']] = m
    branching_factor = len(choices)
    for s in choices:
        #tps2[s['name']] = math.exp(5.0*ts[s['name']])
        tps[s['name']] = math.exp((gamma/branching_factor)*ts[s['name']])
        sis[s['name']] = ts[s['name']]
        d += tps[s['name']]
        #d2 += tps2[s['name']]
    return tps[domain['name']]/d


def get_improvement(init, final):
    imp = 0.0
    for t, p in init.items():
        imp += (final[t] - p)
    return imp


def get_domains_to_update(g, domains, nodes, tagdomains, domainclouds, head, domaintags):

    # build an index on domain names
    domain_index = dict()
    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = []
        domain_index[domains[i]['name']].append(i)

    updomains = []
    dnames = []
    leaves = orgg.get_leaves(g)
    # get the tags in the touched nodes
    leaf_nodes = set(nx.descendants(g, head)).intersection(set(leaves))

    for s in leaf_nodes:
        for d in tagdomains[g.node[s]['tag']]:
            if d['name'] not in dnames:
                if True:
                #if g.node[head][d['name']]['reach_prob_domain'] > 0.0:
                #if node_dom_sims[head][d['name']] > 0.6:
                    updomains.append(d)
                    dnames.append(d['name'])

    # finding intact domains
    intactdomains = []
    for dom in domains:
        if dom['name'] not in dnames and dom['name'] not in intactdomains:
            intactdomains.append(dom['name'])

    update_domains = list(updomains)
    update_domain_names = list(dnames)
    # adding intact domains that have tag cloud in have-to-change domains
    # if a domain has a tag in the have-to-change subtree, it is added to
    # to have-to-change domains
    for dom in intactdomains:
        #add = False
        #for t in domaintags[dom]:
        #    if t in leaf_tags:
        #        add = True
        #if add:
        #    for l in domain_index[dom]:
        #        update_domains.append(domains[l])
        #    update_domain_names.append(dom)

        for dp in list(domainclouds[dom].keys()):
            if dp in update_domain_names:
                continue
            if dp in dnames:
                for l in domain_index[dp]:
                    update_domains.append(domains[l])
                update_domain_names.append(dp)
            #else:
            #    add = False
            #    for t in domaintags[dp]:
            #        if t in leaf_tags:
            #            add = True
            #    if add:
            #        for l in domain_index[dp]:
            #            update_domains.append(domains[l])
            #        update_domain_names.append(dp)


    return update_domains, update_domain_names



def get_dimensions(tags, vecs, n_dims):
    print('vecs: %d tags: %d' % (len(vecs), len(tags)))
    kmeans = KMeans(n_clusters=n_dims, random_state=random.randint(1,1000)).fit(vecs)
    dims = dict()
    dimvecs = dict()
    for i in range(len(kmeans.labels_)):
        c = kmeans.labels_[i]
        if c not in dims:
            dims[c] = []
            dimvecs[c] = []
        dims[c].append(tags[i])
        dimvecs[c].append(vecs[i])
    # computing the means vectors
    cmeans = dict()
    for c, ts in dims.items():
        cmeans[c] = np.mean(dimvecs[c], axis=0)
    # merge the clusters with one member into a random cluster
    ocs = list(dims.keys())
    for c in ocs:
        ts = dims[c]
        print(len(ts))
        if len(ts) > 15:
            continue
        # find the cluster to merge with
        mergec = c
        maxsim = 0.0
        for k in list(dims.keys()):
            if c != k:
                sim = max(0.000001, cosine(cmeans[c], cmeans[k]))
                if sim > maxsim:
                    maxsim = sim
                    mergec = k
        # merge
        print('merging cluster %d with %d.' % (c, mergec))
        dims[mergec].extend(dims[c])
        del dims[c]
        ocs.remove(c)
        print('ocs: %d dims: %d' % (len(ocs), len(dims)))
    print('number of dims: %d' % len(dims))
    return dims

def get_dimensions_plus(tags, vecs):
    print('vecs: %d tags: %d' % (len(vecs), len(tags)))
    range_n_dims = [i for i in range(2, 20)]
    for n_dims in range_n_dims:
        kmeans = KMeans(n_clusters=n_dims, random_state=random.randint(1,1000)).fit(vecs)
        cluster_labels = kmeans.fit_predict(vecs)
        silhouette_avg = silhouette_score(vecs, cluster_labels)
        print("For n_clusters =", n_dims, "The average silhouette_score is :", silhouette_avg)
    dims = dict()
    dimvecs = dict()
    for i in range(len(kmeans.labels_)):
        c = kmeans.labels_[i]
        if c not in dims:
            dims[c] = []
            dimvecs[c] = []
        dims[c].append(tags[i])
        dimvecs[c].append(vecs[i])
    # computing the means vectors
    cmeans = dict()
    for c, ts in dims.items():
        cmeans[c] = np.mean(dimvecs[c], axis=0)
    # merge the clusters with one member into a random cluster
    ocs = list(dims.keys())
    for c in ocs:
        ts = dims[c]
        print(len(ts))
        if len(ts) > 15:
            continue
        # find the cluster to merge with
        mergec = c
        maxsim = 0.0
        for k in list(dims.keys()):
            if c != k:
                sim = max(0.000001, cosine(cmeans[c], cmeans[k]))
                if sim > maxsim:
                    maxsim = sim
                    mergec = k
        # merge
        print('merging cluster %d with %d.' % (c, mergec))
        dims[mergec].extend(dims[c])
        del dims[c]
        ocs.remove(c)
        print('ocs: %d dims: %d' % (len(ocs), len(dims)))
    print('number of dims: %d' % len(dims))
    return dims



def get_dimension_selection_prob(rs, domain):
    global gamma
    d = 0.0
    d2 = 0.0
    tps = dict()
    tps2 = dict()
    sis = dict()
    tsl = []
    ts = dict()
    sps = rs
    branching_factor = len(sps)
    for i in range(len(sps)):
        s = sps[i]
        m = get_transition_sim(s['rep'], domain['mean'])
        tsl.append(m)
        ts[i] = m
    for s in range(len(sps)):
        tps2[s] = math.exp(5.0*ts[s])#(10.0/branching_factor)*ts[s])
        tps[s]  = math.exp((gamma/branching_factor)*ts[s])
        sis[s] = ts[s]
        d += tps[s]
        d2 += tps2[s]
    for s in range(len(sps)):
        tps[s] = tps[s]/d
        tps2[s] = tps2[s]/d2
    ps = [0.0 for i in rs]
    for k, v in tps.items():
        ps[k] = v
    return ps


def save(h, hierarchy_filename):
    hfile = open(hierarchy_filename,'w')
    line = str(len(h.nodes))
    hfile.write(line + '\n')
    for s in h.nodes:
        line = str(s) + ':' + '|'.join(map(str, h.node[s]['tags']))
        hfile.write(line + '\n')

    line = str(len(h.edges))
    hfile.write(line + '\n')
    for e in h.edges:
        line = str(e[0]) + ':' + str(e[1])
        hfile.write(line + '\n')
    hfile.close()
















