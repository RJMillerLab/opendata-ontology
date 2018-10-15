import networkx as nx
import copy
import org.graph as orgg
import numpy as np
import math
import datetime
import random
from sklearn.cluster import KMeans
import json
from sklearn.metrics import silhouette_score

node_dom_sims = dict()
dom_selection_probs = dict()
h = nx.DiGraph()
leaves = []
top = []
gamma = 10.0
domain_index = dict()

def init(g, domains, tagdomains, simfile, tgparam=10.0):
    global domain_index, node_dom_sims, dom_selection_probs, dom_sims, gamma
    gamma = float(tgparam)

    dom_sims = json.load(open(simfile, 'r'))

    print('domains: %d' % len(domains))

    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = i

    #json.dump(tag_dom_sims, open(nodedomsimfile, 'w'))
    print('done init')


def get_state_domain_sims(g, tagdomsimfile, domains):
    print('domains: %d stats: %d' % (len(domains), len(g.nodes)))
    global node_dom_sims
    # Tag-domain sims are precalcualted.
    # Now, the state-domain sims are calculated for the dynamic hierarchy.
    i = 0
    leaves = orgg.get_leaves(g)
    tag_dom_sims = json.load(open(tagdomsimfile, 'r'))
    for l in leaves:
        node_dom_sims[l] = copy.deepcopy(tag_dom_sims[g.node[l]['tag']])
        if len(tag_dom_sims[g.node[l]['tag']]) == 0:
            print('no doms')
    print('loaded %d nodes: leaves: %d' % (len(node_dom_sims), len(leaves)))
    for n in g.nodes:
        i += 1
        if i % 100 == 0:
            print('processed %d states.' % i)
        if n in leaves:
            continue
        node_dom_sims[n] = dict()
        for dom in domains:
            s = get_transition_sim(g.node[n]['rep'],dom['mean'])
            node_dom_sims[n][dom['name']] = s
    print('node_dom_sims: %d' % len(node_dom_sims))




def update_node_dom_sims(g, domains, ns):
    for n in ns:
        for dom in domains:
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])


def add_node_vecs(g, vecs, tags):
    leaves = orgg.get_leaves(g)
    for n in g.nodes:
        if n not in leaves:
            node_vecs = vecs[np.array(list(leaves.intersection(nx.descendants(g,n))))]
            node_tags = np.array(tags)[np.array(list(leaves.intersection(nx.descendants(g,n))))]
            g.node[n]['tags'] = list(set(node_tags))
            g.node[n]['rep'] = np.mean(node_vecs, axis=0)
        else:
            g.node[n]['rep'] = list(vecs[n])
            g.node[n]['tags'] = [tags[n]]
    return g


def get_partial_domain_edge_probs(g, domain, pots, head, leaves, gnodes, to_comp):
    gd = g
    for p in gnodes:
        if p in leaves:
            continue
        #if p not in nodes and p != head:
        #    continue
        if p not in to_comp:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            if ch not in pots:
                print('changed edge has node not in pots')
            #if ch in pots:
            gd[p][ch][domain['name']] = {'trans_prob_domain': prob}
    return gd


def get_sims(g, domain, nodes):
    sims = dict()
    for n in nodes:
        #sims[n] = get_transition_sim(g.node[n]['rep'], domain['mean'])
        sims[n] = node_dom_sims[n][domain['name']]
    return sims


def get_domain_edge_probs(g, domain, leaves, gnodes):
    gd = g
    for p in gnodes:
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            gd[p][ch][domain['name']] = {'trans_prob_domain': prob}
    return gd


def get_partial_domain_node_probs(g, domain, top, pots, head, root, to_comp):
    gd = g
    for n in pots:
        gd.node[n][domain['name']] = dict()
        if n == root:
            gd.node[n][domain['name']]['reach_prob_domain'] = 1.0
        else:
            gd.node[n][domain['name']]['reach_prob_domain'] = 0.0
    for p in top:
        if p not in to_comp:
            continue
        for ch in list(gd.successors(p)):
            if ch not in pots:
                print('visiting node not in pots')
                continue
            gd.node[ch][domain['name']]['reach_prob_domain'] += gd.node[p][domain['name']]['reach_prob_domain']*                                       gd[p][ch][domain['name']]['trans_prob_domain']
    return gd



def get_domain_node_probs(gd, domain, top, root, gnodes):

    domainname = domain['name']
    for n in gnodes:
        gd.node[n][domainname] = dict()
        if n == root:
            gd.node[n][domainname]['reach_prob_domain'] = 1.0
        else:
            gd.node[n][domainname]['reach_prob_domain'] = 0.0
    for p in top:
        for ch in list(gd.successors(p)):
            gd.node[ch][domainname]['reach_prob_domain'] += gd.node[p][domainname]['reach_prob_domain']*gd[p][ch][domainname]['trans_prob_domain']
    return gd



def get_domain_node_probs_plus(gd, domain, top, root, gnodes):
    for n in gnodes:
        gd.node[n][domain['name']] = dict()
        if n == root:
            gd.node[n][domain['name']]['reach_prob_domain'] = 1.0
        else:
            gd.node[n][domain['name']]['reach_prob_domain'] = 0.0
    for p in top:
        for ch in list(gd.successors(p)):
            gd.node[ch][domain['name']]['reach_prob_domain'] += gd.node[p][domain['name']]['reach_prob_domain']*                                       gd[p][ch][domain['name']]['trans_prob_domain']
    return gd



def get_transition_sim(vec1, vec2):
    # cosine similarity
    c = max(0.000001, cosine(vec1, vec2))
    return c


def cosine(vec1, vec2):
    dp = np.dot(vec1,vec2)
    c = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return c


def get_success_prob_likelihood_partial(g, adomains, tagdomains, domainclouds, dtype, domaintags, nodes, update_head,  prev_success_probs, prev_domain_success_probs):

    active_domains, dnames = get_domains_to_update(g.copy(), adomains, nodes, tagdomains, domainclouds, update_head, domaintags, False)
    print('exact doms: %d vs. %d' % (len(active_domains), len(adomains)))

    start = datetime.datetime.now()
    expected_success1, h1, success_probs1, likelihood1, domain_success_probs1 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, dnames, update_head, nodes, False)
    e3 = datetime.datetime.now()-start


    print('dom exact: %d' % (int(e3.total_seconds() *1000)))


    return expected_success1, h1.copy(), success_probs1, likelihood1, domain_success_probs1, len(active_domains)



def get_success_prob_likelihood_partial_plus(g, adomains, tagdomains, domainclouds, dtype, domaintags, nodes, update_head,  prev_success_probs, prev_domain_success_probs):

    gnodes_num = len(g.nodes)

    active_domains, dnames = get_domains_to_update(g.copy(), adomains, nodes, tagdomains, domainclouds, update_head, domaintags, False)
    print('exact doms: %d vs. %d' % (len(active_domains), len(adomains)))
    to_comp = list(set(sum((list(g.predecessors(n)) for n in nodes), [])))
    print('visited %d nodes instead of %d' % (len(to_comp), gnodes_num))

    sts = []
    for n in nodes:
        sts.extend(g.node[n]['tags'])
    sts=set(sts)

    start = datetime.datetime.now()
    expected_success4, h4, success_probs4, likelihood4, domain_success_probs4 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, dnames, update_head, nodes, False)
    e3 = datetime.datetime.now()-start


    start = datetime.datetime.now()
    expected_success1, h1, success_probs1, likelihood1, domain_success_probs1 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, dnames, update_head, nodes, True)
    e1 = datetime.datetime.now()-start


    start = datetime.datetime.now()
    expected_success3, h3, success_probs3, likelihood3, domain_success_probs3 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype,      domaintags, prev_success_probs, prev_domain_success_probs, adomains, [dom['name'] for dom in adomains], update_head, nodes, False)
    e2 = datetime.datetime.now()-start


    print('elapsed time of node+dom %d dom %d exact %d' % (int(e1.total_seconds() *1000), int(e3.total_seconds() *1000), int(e2.total_seconds() *1000)))



    if set(nodes) != set(nx.descendants(h1, update_head)):
        print('err: nodes and others not equal %d %d' % (len(set(nodes)), len(set(nx.descendants(h1, update_head)))))

    if not math.isclose(expected_success3, expected_success1, rel_tol=1e-5):
        print('unmatched dsp: %d' % len([d for d, p in domain_success_probs3.items() if not math.isclose(p,           domain_success_probs1[d], rel_tol=1e-5)]))
        print('unmtached dsp in dnames: %d' % len([d for d, p in domain_success_probs3.items() if d in dnames and not math.isclose(p,           domain_success_probs1[d], rel_tol=1e-5)]))
        print('unmtached dsp not in dnames: %d' % len([d for d, p in                         domain_success_probs3.items() if d not in dnames and not math.isclose(p,                     domain_success_probs1[d], rel_tol=1e-5)]))
        print('err: mistake in computing success prob: exact: %f and prune: %f' % (expected_success3, expected_success1))
        for d, p in domain_success_probs3.items():
            if not math.isclose(p, domain_success_probs1[d], rel_tol=1e-5):
                if d not in dnames:
                    if not math.isclose(prev_domain_success_probs[d], domain_success_probs3[d], rel_tol=1e-5):
                        print('dom should have been examined')
                for n in h1.nodes:
                    if not math.isclose(h1.node[n][d]['reach_prob_domain'], h3.node[n][d]['reach_prob_domain'], rel_tol=1e-5):
                        #
                        if n not in nodes:
                            print('not in nodes')
                        for e in h1.predecessors(n):
                            if d not in h1[e][n]:
                                if d in dnames:
                                    print('edge has not been visited')
                                    if e == update_head:
                                        print('added edge has not been visited')
                                continue
                            if not math.isclose(h1[e][n][d]['trans_prob_domain'],h3[e][n][d]['trans_prob_domain'], rel_tol=1e-5):
                                if math.isclose(h3[e][n][d]['trans_prob_domain'], g[e][n][d]['trans_prob_domain'], rel_tol=1e-5):
                                    print('edge should not have been updated')
                                elif math.isclose(h1[e][n][d]['trans_prob_domain'],g[e][n][d]['trans_prob_domain'], rel_tol=1e-5):
                                    if e not in nodes:
                                        print('edge from non potential node should have been updated')
                                    else:
                                        print('edge should have been updated')
                                else:
                                    print('edge updated incorrectly')
                            #

    return expected_success1, h1.copy(), success_probs1, likelihood1, domain_success_probs1


def common_node_tags_domain(h, n, dts):
    if len(dts) ==0:
        print('err: no domain tags')

    leaves = orgg.get_leaves(h)
    ls={n}
    if n not in leaves:
        ls = leaves.intersection(nx.descendants(h, n))
    ts = []
    for l in ls:
        ts.append(h.node[l]['tag'])
    if len(ts) ==0:
        print('err: len(ts) is zero')
    if len(h.node[n]['tags'])==0:
        print('err: len(h.node[n][tags]) is zero')
    if set(ts) != set(h.node[n]['tags']):
        print('err: tags do not match: %d %d' % (len(set(ts)), len(set(h.node[n]['tags']))))
    if len(set(ts).intersection(set(dts))) == 0:
        return False
    return True


def get_success_prob_prune_domains(g, adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, update_domain_names, head, potentials, prune):
    print('started get_success_prob_prune_domains')

    domains = list(active_domains)

    h = g.copy()
    top = list(nx.topological_sort(g))
    #gnodes = orgg.get_nodes(g)
    gnodes = list(g.nodes)

    success_probs = dict()
    success_probs_intersect = dict()
    domain_success_probs = dict()
    likelihood = 0.0

    root = orgg.get_root_plus(g, gnodes)
    leaves = orgg.get_leaves_plus(g, gnodes)

    # reset states' reachability
    for p in gnodes:
        h.node[p]['reach_prob'] = 0
    # the reachability prob of domains that are not candidates for update remains intact
    for dom in adomains:
        if dom['name'] in update_domain_names:
            continue

        table = get_table_from_domain(dom['name'], dtype)

        sp = prev_domain_success_probs[dom['name']]
        if table not in success_probs_intersect:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp
        domain_success_probs[dom['name']] = prev_domain_success_probs[dom['name']]
        # accumulate the reachability of non-updatable domains
        for p in gnodes:
            h.node[p]['reach_prob'] += h.node[p][dom['name']]['reach_prob_domain']


    to_comp = list(set(sum((list(g.predecessors(n)) for n in potentials), [])))
    samedom = 0
    dom_target_sims = []
    reachable_dom_probs = []
    #
    inc_count = 0
    dec_count = 0
    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        accepted_tags.extend(list(domaintags[domain['name']]))
        for c in list(domainclouds[domain['name']].keys()):
            if c not in domain_index:
                #print('domain cloud not in index.')
                continue
            accepted_tags.extend(domaintags[adomains[domain_index[c]]['name']])
        accepted_tags = list(set(accepted_tags))
        #
        table = get_table_from_domain(domain['name'], dtype)

        if prune:
            gp = get_partial_domain_edge_probs(h, domain, potentials, head, leaves, gnodes, to_comp)
            gpp = get_partial_domain_node_probs(gp, domain, top, potentials, head, root, to_comp)
        else:
            gp = get_domain_edge_probs(h, domain, leaves, gnodes)
            gpp = get_domain_node_probs(gp, domain, top, root, gnodes)
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

        for p in gnodes:
        #for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']

        h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0

    for p in gnodes:
    #for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(adomains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    print('inc: %d dec: %d out of %d (%d)' % (inc_count, dec_count, len(domains), len(adomains)))

    return expected_success, h, success_probs, likelihood, domain_success_probs



def get_success_prob_likelihood_fuzzy(g, domains, tagdomains, domainclouds, dtype, domaintags):

    top = list(nx.topological_sort(g))
    gnodes = list(g.nodes)
    root = orgg.get_root_plus(g, gnodes)
    leaves = orgg.get_leaves_plus(g, gnodes)
    success_probs = dict()
    domain_success_probs = dict()
    success_probs_intersect = dict()
    h = g
    likelihood = 0.0
    samedom = 0
    dom_target_sims = []
    reachable_dom_probs = []
    #
    for p in gnodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        accepted_tags.extend(list(domaintags[domain['name']]))
        for c in list(domainclouds[domain['name']].keys()):
            if c not in domain_index:
                #print('domain cloud not in index.')
                continue
            accepted_tags.extend(domaintags[domains[domain_index[c]]['name']])
        accepted_tags = list(set(accepted_tags))
        #
        table = get_table_from_domain(domain['name'], dtype)

        gp = get_domain_edge_probs(h, domain, leaves, gnodes)
        gpp = get_domain_node_probs(gp, domain, top, root, gnodes)
        # finding the most reachable domain
        max_reached_dom_prob = 0.0
        max_reached_dom_sim = 0.0
        most_reachable_dom = ''
        #
        for n in leaves:
            stag = gpp.node[n]['tag']
            if stag not in accepted_tags:
                continue
            selps, sims = get_dom_trans_prob(tagdomains[stag], domain)
            for d in tagdomains[stag]:
                if d['name'] not in domainclouds[domain['name']]:
                    continue
                sp = selps[d['name']] * gpp.node[n][domain['name']]['reach_prob_domain']
                if sp > max_reached_dom_prob:
                    max_reached_dom_prob = sp
                    max_reached_dom_sim = sims[d['name']]
                    most_reachable_dom = d['name']
        dom_target_sims.append(max_reached_dom_sim)
        reachable_dom_probs.append(max_reached_dom_prob)

        table = get_table_from_domain(domain['name'], dtype)

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

        for p in gnodes:
        #for p in gpp.nodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domain['name']]['reach_prob_domain']

        h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0

    for p in gnodes:
    #for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    return expected_success, h, success_probs, likelihood, domain_success_probs


def get_dom_trans_prob(choices, domain):
    global gamma
    d2 = 0.0
    tps2 = dict()
    sis = dict()
    branching_factor = len(choices)
    for s in choices:
        #m = get_transition_sim(s['mean'], domain['mean'])
        m = 0.0
        if s['name'] in dom_sims:
            if domain['name'] in dom_sims[s['name']]:
                m = dom_sims[s['name']][domain['name']]
        tps2[s['name']] = math.exp((gamma/branching_factor)*m)
        sis[s['name']] = m
        d2 += tps2[s['name']]
    for s in choices:
        tps2[s['name']] = (tps2[s['name']]/d2)
    return tps2, sis




def get_trans_prob(g, p, domain):
    global gamma
    d = 0.0
    tps = dict()
    sis = dict()
    sps = list(g.successors(p))
    branching_factor = len(sps)
    for s in sps:
        #m = get_transition_sim(g.node[s]['rep'], domain['mean'])
        m = node_dom_sims[s][domain['name']]
        sis[s] = m
        tps[s] = math.exp((gamma/branching_factor)*m)
        d += tps[s]
    for s in sps:
        tps[s] = tps[s]/d

    return tps, sis


def get_domains_to_update(g, domains, nodes, tagdomains, domainclouds, head, domaintags, prune):

    sts = []
    for n in nodes:
        sts.extend(g.node[n]['tags'])
    sts=set(sts)

    updomains = []
    dnames = []
    leaves = orgg.get_leaves(g)
    # get the tags in the touched nodes
    leaf_nodes = [head]
    if head not in leaves:
        leaf_nodes = list(set(list(nx.descendants(g, head))).intersection(set(leaves)))

    for s in leaf_nodes:
        if len(tagdomains[g.node[s]['tag']]) == 0:
            print('err: no dom found')
        for d in tagdomains[g.node[s]['tag']]:
            if d['name'] not in dnames:
                if not prune:
                #if g.node[head][d['name']]['reach_prob_domain'] > 0.0:
                #if node_dom_sims[head][d['name']] > 0.6:
                    updomains.append(d)
                    dnames.append(d['name'])
                elif node_dom_sims[head][d['name']] > 0.6:
                    updomains.append(d)
                    dnames.append(d['name'])

    if set([g.node[l]['tag'] for l in leaf_nodes])!=sts:
        print('err: tags in nodes are diff %d vs %d' % (len(set([g.node[l]['tag'] for l in leaf_nodes])), len(sts)))


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

        if dom in update_domain_names:
            continue
        for dp in list(domainclouds[dom].keys()):
            if dp in dnames and dom not in update_domain_names:
                update_domains.append(domains[domain_index[dom]])
                update_domain_names.append(dom)

    if len(update_domain_names) > len(dnames):
        print('added domains from c prime.')


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



def save(h, hierarchy_filename):
    hfile = open(hierarchy_filename,'w')
    hnodes = list(h.nodes)
    line = str(len(hnodes))
    hfile.write(line + '\n')
    for s in hnodes:
        line = str(s) + ':' + '|'.join(map(str, h.node[s]['tags']))
        hfile.write(line + '\n')

    line = str(len(h.edges))
    hfile.write(line + '\n')
    for e in h.edges:
        line = str(e[0]) + ':' + str(e[1])
        hfile.write(line + '\n')
    hfile.close()


def get_tag_domain_sim(domains, tags, vecs, tagdomsimfile):
    print('get_tag_domain_sim')
    sims = dict()
    for i in range(len(tags)):
        if i % 100 == 0:
            print('computed tag dom sims for %d tags' % i)
        t = tags[i]
        v = vecs[i]
        if t not in sims:
            sims[t] = dict()
        for d in domains:
            s = get_transition_sim(v,d['mean'])
            sims[t][d['name']] = s
    json.dump(sims, open(tagdomsimfile, 'w'))
    print('done computing sims for %d tags ' % (len(sims)))



def get_table_from_domain(domainname, dtype):
    table = ''
    if dtype == 'synthetic':
        colid = int(domainname[domainname.rfind('_')+1:])
        table = domainname[:domainname.rfind('_')]+'_'+str(colid%2)
    if dtype == 'opendata':
        table = domainname[:domainname.rfind('_')]
    return table


