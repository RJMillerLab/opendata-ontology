import networkx as nx
import org.graph as orgg
import operator
import numpy as np
import math
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
    #print('trans sim > 0.5')
    #print('reach prob > 0.1')
    print('sanity check of domain prunning')
    global domain_index, node_dom_sims, dom_selection_probs, dom_sims, gamma
    gamma = float(tgparam)
    print('gamma: %f' % gamma)

    dom_sims = json.load(open(simfile, 'r'))

    #dom_selection_probs = get_domains_selection_probs(tagdomains)

    node_dom_sims = {n:{dom['name']: get_transition_sim(g.node[n]['rep'], dom['mean']) for dom in domains} for n in g.nodes}

    #for n in g.nodes:
    #    node_dom_sims[n] = dict()
    #    for dom in domains:
    #        node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])

    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = []
        domain_index[domains[i]['name']].append(i)

    print('done init')


def update_node_dom_sims(g, domains, ns):
    for n in ns:
        for dom in domains:
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])



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
            g.node[n]['tags'] = list(set(node_tags))
            g.node[n]['rep'] = np.mean(node_vecs, axis=0)
            #g.node[n]['cov'] = np.cov(node_vecs)
            g.node[n]['mean'] = g.node[n]['rep']
            #g.node[n]['det'] = linalg.det(g.node[n]['cov'])
        else:
            g.node[n]['rep'] = list(vecs[n])
            g.node[n]['population'] = [vecs[n]]
            g.node[n]['tags'] = [tags[n]]
    return g



def get_tag_probs(g, domain):
    tags = dict()
    for n in orgg.get_leaves(g):
        tags[g.node[n]['tag']] = g.node[n][domain['name']]['reach_prob_domain']
    return tags


def get_sims(g, domain, nodes):
    sims = dict()
    for n in nodes:
        #sims[n] = get_transition_sim(g.node[n]['rep'], domain['mean'])
        sims[n] = node_dom_sims[n][domain['name']]
    return sims


def get_domain_edge_probs(g, domain, leaves, gnodes, gedges):
    gd = g
    for p in list(gnodes.keys()):
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            gedges[p][ch][domain['name']] = {'trans_prob_domain': prob, 'trans_sim_domain': sis[ch]}
    return gd, gedges


def get_domain_node_probs(g, domain, top, root, gnodes, gedges):
    gd = g
    for n in list(gnodes.keys()):
        gnodes[n][domain['name']] = dict()
        if n == root:
            gnodes[n][domain['name']]['reach_prob_domain'] = 1.0

        else:
            gnodes[n][domain['name']]['reach_prob_domain'] = 0.0

    for p in top:
        for ch in list(gd.successors(p)):
            gnodes[ch][domain['name']]['reach_prob_domain'] += gnodes[p][domain['name']]['reach_prob_domain']*                                       gedges[p][ch][domain['name']]['trans_prob_domain']
    return gd, gnodes, gedges


def get_transition_sim(vec1, vec2):
    # cosine similarity
    c = max(0.000001, cosine(vec1, vec2))
    return c


def cosine(vec1, vec2):
    dp = np.dot(vec1,vec2)
    c = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return c



def get_success_prob_likelihood_partial(g, adomains, tagdomains, domainclouds, dtype, domaintags, nodes, update_head,  prev_success_probs, prev_domain_success_probs):


    active_domains, dnames = get_domains_to_update(g.copy(), adomains, nodes, tagdomains, domainclouds, update_head, domaintags)
    print('exact doms: %d vs. %d' % (len(active_domains), len(adomains)))

    expected_success1, h1, success_probs1, likelihood1, domain_success_probs1 = get_success_prob_prune_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, dnames)

    return expected_success1, h1.copy(), success_probs1, likelihood1, domain_success_probs1


def get_success_prob_prune_domains(g, adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_domains, update_domain_names):

    domains = list(active_domains)

    h = g.copy()
    top = list(nx.topological_sort(g))
    root = orgg.get_root(g)
    gnodes = orgg.get_nodes(g)
    gedges = orgg.get_edges(g)
    print('got nodes and edges from org.')

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
    #
    inc_count = 0
    dec_count = 0
    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        accepted_tags.extend(list(domaintags[domain['name']]))
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
        gp, gedges = get_domain_edge_probs(h, domain, leaves, gnodes, gedges)
        gpp, gnodes, gedges = get_domain_node_probs(gp, domain, top, root, gnodes, gedges)

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

    gp = orgg.update_edges_from_dict(h, gedges)
    h = orgg.update_nodes_from_dict(gp, gnodes)


    for p in h.nodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(adomains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    print('inc: %d dec: %d out of %d (%d)' % (inc_count, dec_count, len(domains), len(adomains)))

    return expected_success, h, success_probs, likelihood, domain_success_probs




def get_success_prob_likelihood_fuzzy(g, domains, tagdomains, domainclouds, dtype, domaintags):
    print('get_success_prob_likelihood_fuzzy')

    top = list(nx.topological_sort(g))
    root = orgg.get_root(g)
    gnodes = orgg.get_nodes(g)
    gedges = orgg.get_edges(g)
    print('loaded gnodes and gedges')
    success_probs = dict()
    domain_success_probs = dict()
    success_probs_intersect = dict()
    h = g
    leaves = orgg.get_leaves(g)
    likelihood = 0.0
    samedom = 0
    dom_target_sims = []
    reachable_dom_probs = []
    #
    for p in h.nodes:
        h.node[p]['reach_prob'] = 0.0

    print('started doms')
    for domain in domains:
        # finding the tags of accepted domains
        accepted_tags = []
        accepted_tags.extend(list(domaintags[domain['name']]))
        for c in list(domainclouds[domain['name']].keys()):
            if c not in domain_index:
                continue
            for di in domain_index[c]:
                accepted_tags.extend(domaintags[domains[di]['name']])
        accepted_tags = list(set(accepted_tags))
        #
        table = ''
        if dtype == 'synthetic':
            colid = int(domain['name'][domain['name'].rfind('_')+1:])
            table = domain['name'][:domain['name'].rfind('_')]+'_'+str(colid%2)
        if dtype == 'opendata':
            table = domain['name'][:domain['name'].rfind('_')]
        gp, gedges = get_domain_edge_probs(h, domain, leaves, gnodes, gedges)
        gpp, gnodes, gedges = get_domain_node_probs(gp, domain, top, root, gnodes, gedges)
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
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0

    gp = orgg.update_nodes_from_dict(h, gnodes)
    h = orgg.update_edges_from_dict(gp, gedges)

    for p in h.nodes:
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
    tps2 = {s['name']:(tps2[s['name']]/d2) for s in choices}
    #for s in choices:
    #    tps2[s['name']] = (tps2[s['name']]/d2)
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
    sps = {s:tps[s]/d for s in sps}
    #for s in sps:
    #    tps[s] = tps[s]/d

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
        tps[s['name']] = math.exp((gamma/branching_factor)*ts[s['name']])
        sis[s['name']] = ts[s['name']]
        d += tps[s['name']]
    return tps[domain['name']]/d



def get_domains_to_update(g, domains, nodes, tagdomains, domainclouds, head, domaintags):

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
                #if True:
                #if g.node[head][d['name']]['reach_prob_domain'] > 0.0:
                if node_dom_sims[head][d['name']] > 0.6:
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
                for l in domain_index[dom]:
                    update_domains.append(domains[l])
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





