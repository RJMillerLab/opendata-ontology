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

def init(g, domainsfile, simfile, tgparam=10.0):
    global domain_index, dom_sims, gamma
    gamma = float(tgparam)

    domain_index = dict()

    domains = json.load(open(domainsfile, 'r'))

    dom_sims = json.load(open(simfile, 'r'))

    print('org.init domains: %d' % len(domains))

    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = i
        else:
            print('dup domains')


    #json.dump(tag_dom_sims, open(nodedomsimfile, 'w'))
    print('done init')


def get_state_domain_sims(g, tagdomsimfile, domains):
    print('domains: %d states: %d' % (len(domains), len(g.nodes)))
    global node_dom_sims
    node_dom_sims = dict()
    # Tag-domain sims are precalcualted.
    # Now, the state-domain sims are calculated for the dynamic hierarchy.
    i = 0
    leaves = orgg.get_leaves(g)
    tag_dom_sims = json.load(open(tagdomsimfile, 'r'))
    for l in leaves:
        node_dom_sims[l] = copy.deepcopy(tag_dom_sims[g.node[l]['tag']])
        if len(tag_dom_sims[g.node[l]['tag']]) == 0:
            print('no doms')
    print('loaded %d nodes and leaves: %d' % (len(node_dom_sims), len(leaves)))
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
    print('node_dom_sims: %d and states: %d' % (len(node_dom_sims), len(g.nodes)))




def update_node_dom_sims(g, domains, ns):
    for n in ns:
        for dom in domains:
            node_dom_sims[n][dom['name']] = get_transition_sim(g.node[n]['rep'], dom['mean'])

def extend_state_dom_sims(g, domains):
    for i in range(len(domains)):
        if i%100 == 0:
            print('updated state dom sim for %d domains.' % i)
        dom = domains[i]
        for n in g.nodes:
            if dom['name'] in node_dom_sims[n]:
                print('node_dom_sims dom exists ')
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



def get_sims(g, domain, nodes):
    sims = dict()
    for n in nodes:
        #sims[n] = get_transition_sim(g.node[n]['rep'], domain['mean'])
        sims[n] = node_dom_sims[n][domain['name']]
    return sims


def get_rep_edge_probs(g, domain, leaves, gnodes):
    gd = g
    for p in gnodes:
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            gd[p][ch][domain['name']] = {'trans_prob_domain': prob}
    return gd



def get_rep_node_probs(gd, domain, top, root, gnodes):

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




def get_transition_sim(vec1, vec2):
    # cosine similarity
    c = max(0.000001, cosine(vec1, vec2))
    return c


def cosine(vec1, vec2):
    dp = np.dot(vec1,vec2)
    c = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return c


def get_success_prob_rep_partial(g, adomains, tagdomains, domainclouds, dtype, domaintags, nodes, update_head,  prev_success_probs, prev_domain_success_probs, repdomains, reps):

    start = datetime.datetime.now()
    active_reps, drnames, active_domain_names = get_reps_to_update(g.copy(), adomains, nodes, tagdomains, domainclouds, update_head, domaintags, False, repdomains, reps)
    print('finding active doms: %d' % (int((datetime.datetime.now()-start).total_seconds() *1000)))
    print('exact dom prunning: %d  rep: %d' % (len(active_domain_names), len(active_reps)))

    start = datetime.datetime.now()

    expected_success1, h1, success_probs1, likelihood1, domain_success_probs1 = get_success_prob_rep_domains(g.copy(), adomains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_reps, drnames, update_head, nodes, False, repdomains, reps)
    e3 = datetime.datetime.now()-start


    print('dom exact: %d' % (int(e3.total_seconds() *1000)))


    return expected_success1, h1.copy(), success_probs1, likelihood1, domain_success_probs1, len(active_domain_names), len(active_reps)



# approximating domain sps with their corresponding rep sps.
def get_success_prob_rep_domains(g, domains, tagdomains, domainclouds, dtype, domaintags, prev_success_probs, prev_domain_success_probs, active_reps, active_rep_names, head, potentials, prune, repdomains, reps):
    print('started get_success_prob_prune_domains')

    h = g.copy()
    top = list(nx.topological_sort(g))
    gnodes = list(g.nodes)

    success_probs = dict()
    success_probs_intersect = dict()
    domain_success_probs = dict()
    likelihood = 0.0

    root = orgg.get_root_plus(g, gnodes)
    leaves = orgg.get_leaves_plus(g, gnodes)

    # the reachability prob of reps that are not candidates for update remains intact
    for rep in reps:
        # dom is a rep domain
        repname = rep['name']
        if repname in active_rep_names:
            continue

        for domname in repdomains[repname]:
            #if domname not in domain_index:
            #    continue

            table = get_table_from_domain(domname, dtype)

            sp = prev_domain_success_probs[domname]
            if table not in success_probs_intersect:
                success_probs_intersect[table] = (1.0-sp)
                success_probs[table] = sp
            else:
                success_probs_intersect[table] *= (1.0-sp)
                success_probs[table] += sp
            domain_success_probs[domname] = prev_domain_success_probs[domname]


    reachable_dom_probs = []
    inc_count = 0
    dec_count = 0
    print('domain_index: %d' % len(domain_index))
    for rep in reps:
        repname = rep['name']

        if repname not in active_rep_names:
            continue

        gp = get_rep_edge_probs(h, rep, leaves, gnodes)
        gpp = get_rep_node_probs(gp, rep, top, root, gnodes)

        for domainname in repdomains[repname]:
            #if domname not in domain_index:
            #    continue
            domain = domains[domain_index[domainname]]

            # finding the tags of accepted domains
            accepted_tags = list(domaintags[domainname])
            for c in list(domainclouds[domainname].keys()):
                if c not in domain_index:
                    print('domain cloud not in index.')
                    continue
                accepted_tags.extend(domaintags[domains[domain_index[c]]['name']])
            accepted_tags = list(set(accepted_tags))
            #
            table = get_table_from_domain(domainname, dtype)

            # finding the most reachable domain
            max_reached_dom_prob = 0.0
            #
            for n in leaves:
                if gpp.node[n]['tag'] not in accepted_tags:
                    continue
                selps = get_dom_trans_prob(tagdomains[gpp.node[n]['tag']], domain)
                for dn in tagdomains[gpp.node[n]['tag']]:
                    if dn not in domainclouds[domainname]:
                        continue
                    sp = selps[dn] * gpp.node[n][repname]['reach_prob_domain']
                    if sp > max_reached_dom_prob:
                        max_reached_dom_prob = sp
            reachable_dom_probs.append(max_reached_dom_prob)
            sp = max_reached_dom_prob

            if domainname in domain_success_probs:
                print('rep dom alert')
            domain_success_probs[domainname] = sp
            if domain_success_probs[domainname] > prev_domain_success_probs[domainname]:
                inc_count += 1
            else:
                dec_count += 1

            if table not in success_probs:
                success_probs_intersect[table] = (1.0-sp)
                success_probs[table] = sp
            else:
                success_probs_intersect[table] *= (1.0-sp)
                success_probs[table] += sp


        h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0


    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    print('inc: %d dec: %d out of %d (%d)' % (inc_count, dec_count, len(domains), len(domains)))

    return expected_success, h, success_probs, likelihood, domain_success_probs



def get_success_rep_prob_fuzzy(g, domains, tagdomains, domainclouds, dtype, domaintags, repdomains, reps):

    print('get_success_rep_prob_fuzzy')

    top = list(nx.topological_sort(g))
    gnodes = list(g.nodes)
    root = orgg.get_root_plus(g, gnodes)
    leaves = orgg.get_leaves_plus(g, gnodes)
    success_probs = dict()
    domain_success_probs = dict()
    success_probs_intersect = dict()
    h = g
    likelihood = 0.0
    #
    for p in gnodes:
        h.node[p]['reach_prob'] = 0.0

    for rep in reps:

        repname = rep['name']

        gp = get_rep_edge_probs(h, rep, leaves, gnodes)
        gpp = get_rep_node_probs(gp, rep, top, root, gnodes)

        for domainname in repdomains[repname]:
            # index is built on domain of a dimension which is a subset
            # of complete domains.
            if domainname not in domain_index:
                continue
            domain = domains[domain_index[domainname]]
            # finding the tags of accepted domains
            accepted_tags = []
            accepted_tags.extend(list(domaintags[domainname]))
            for c in list(domainclouds[domainname].keys()):
                if c not in domain_index:
                    print('domain cloud not in index.')
                    continue
                accepted_tags.extend(domaintags[domains[domain_index[c]]['name']])
            accepted_tags = list(set(accepted_tags))
            #
            table = get_table_from_domain(domainname, dtype)

            # finding the most reachable domain
            max_reached_dom_prob = 0.0
            #
            for n in leaves:
                stag = gpp.node[n]['tag']
                if stag not in accepted_tags:
                    continue
                selps = get_dom_trans_prob(tagdomains[stag], domain)
                for dn in tagdomains[stag]:
                    if dn not in domainclouds[domainname]:
                        continue
                    sp = selps[dn] * gpp.node[n][repname]['reach_prob_domain']
                    if sp > max_reached_dom_prob:
                        max_reached_dom_prob = sp

            sp = max_reached_dom_prob

            domain_success_probs[domainname] = sp


            if table not in success_probs:
                success_probs_intersect[table] = (1.0-sp)
                success_probs[table] = sp
            else:
                success_probs_intersect[table] *= (1.0-sp)
                success_probs[table] += sp

            for p in gnodes:
                gpp.node[p]['reach_prob'] += gpp.node[p][repname]['reach_prob_domain']

            h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0

    for p in gnodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    return expected_success, h, success_probs, likelihood, domain_success_probs


def get_dom_trans_prob(choices, domain):
    global gamma
    d2 = 0.0
    tps2 = dict()
    branching_factor = len(choices)
    for s in choices:
        #m = get_transition_sim(s['mean'], domain['mean'])
        m = 0.0
        if s in dom_sims:
            if domain['name'] in dom_sims[s]:
                m = dom_sims[s][domain['name']]
        tps2[s] = math.exp((gamma/branching_factor)*m)
        d2 += tps2[s]
    for s in choices:
        tps2[s] = (tps2[s]/d2)
    return tps2




def get_trans_prob(g, p, domain):
    global gamma
    d = 0.0
    tps = dict()
    sis = dict()
    sps = list(g.successors(p))
    branching_factor = len(sps)
    for s in sps:
        #m = get_transition_sim(g.node[s]['rep'], domain['mean'])
        m = 0.0
        if domain['name'] in node_dom_sims[s]:
            m = node_dom_sims[s][domain['name']]
        sis[s] = m
        tps[s] = math.exp((gamma/branching_factor)*m)
        d += tps[s]
    for s in sps:
        tps[s] = tps[s]/d

    return tps, sis


def get_reps_to_update(g, domains, nodes, tagdomains, domainclouds, head, domaintags, prune, repdomains, reps):


    updomains = []
    dnames = {}
    leaves = orgg.get_leaves(g)
    # get the tags in the touched nodes
    leaf_nodes = [head]
    if head not in leaves:
        leaf_nodes = list(set(list(nx.descendants(g, head))).intersection(set(leaves)))

    for s in leaf_nodes:
        for dn in tagdomains[g.node[s]['tag']]:
            d = domains[domain_index[dn]]
            if d['name'] not in dnames:
                updomains.append(d)
                dnames[d['name']] = True


    # finding intact domains
    intactdomains = {}
    for dom in domains:
        if dom['name'] not in dnames and dom['name'] not in intactdomains:
            intactdomains[dom['name']] = True

    update_domains = list(updomains)
    update_domain_names = copy.deepcopy(dnames)
    # adding intact domains that have tag cloud in have-to-change domains
    # if a domain has a tag in the have-to-change subtree, it is added to
    # to have-to-change domains
    for dom in list(intactdomains.keys()):
        if dom in update_domain_names:
            continue
        for dp in list(domainclouds[dom].keys()):
            if dp in dnames and dom not in update_domain_names:
                update_domains.append(domains[domain_index[dom]])
                update_domain_names[dom] = True


    active_reps = []
    active_rep_names = dict()

    rep_index = dict()
    for i in range(len(reps)):
        rep_index[reps[i]['name']] = i

    for rep, rdomains in repdomains.items():
        for dom in rdomains:
            if dom in update_domain_names:
                if rep not in active_rep_names:
                    active_rep_names[rep] = True
                    active_reps.append(reps[rep_index[rep]])

    return active_reps, active_rep_names, update_domain_names


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




def get_domains_to_update(g, domains, nodes, tagdomains, domainclouds, head, domaintags, prune):
    if prune:
        print('pruning domains by threshold.')


    updomains = []
    dnames = {}
    leaves = orgg.get_leaves(g)
    # get the tags in the touched nodes
    leaf_nodes = [head]
    if head not in leaves:
        leaf_nodes = list(set(list(nx.descendants(g, head))).intersection(set(leaves)))

    for s in leaf_nodes:
        for dn in tagdomains[g.node[s]['tag']]:
            d = domains[domain_index[dn]]
            if d['name'] not in dnames:
                if not prune:
                    updomains.append(d)
                    dnames[d['name']] = True
                elif node_dom_sims[head][d['name']] > 0.6:
                    updomains.append(d)
                    dnames[d['name']] = True


    # finding intact domains
    intactdomains = {}
    for dom in domains:
        if dom['name'] not in dnames and dom['name'] not in intactdomains:
            intactdomains[dom['name']] = True

    update_domains = list(updomains)
    update_domain_names = copy.deepcopy(dnames)

    # adding intact domains that have tag cloud in have-to-change domains
    # if a domain has a tag in the have-to-change subtree, it is added to
    # to have-to-change domains
    for dom in list(intactdomains.keys()):
        if dom in update_domain_names:
            continue
        for dp in list(domainclouds[dom].keys()):
            if dp in dnames and dom not in update_domain_names:
                update_domains.append(domains[domain_index[dom]])
                update_domain_names[dom] = True


    return update_domains, update_domain_names


def get_success_prob_fuzzy(g, domains, tagdomains, domainclouds, dtype, domaintags):
    print('get_success_prob_fuzzy')

    top = list(nx.topological_sort(g))
    gnodes = list(g.nodes)
    root = orgg.get_root_plus(g, gnodes)
    leaves = orgg.get_leaves_plus(g, gnodes)
    success_probs = dict()
    domain_success_probs = dict()
    success_probs_intersect = dict()
    h = g
    likelihood = 0.0
    reachable_dom_probs = []
    #
    for p in gnodes:
        h.node[p]['reach_prob'] = 0.0

    for domain in domains:
        domainname = domain['name']
        # finding the tags of accepted domains
        accepted_tags = []
        accepted_tags.extend(list(domaintags[domainname]))
        for c in list(domainclouds[domainname].keys()):
            if c not in domain_index:
                print('domain cloud not in index.')
                continue
            accepted_tags.extend(domaintags[domains[domain_index[c]]['name']])
        accepted_tags = list(set(accepted_tags))
        #
        table = get_table_from_domain(domainname, dtype)

        gp = get_domain_edge_probs(h, domain, leaves, gnodes)
        gpp = get_domain_node_probs(gp, domain, top, root, gnodes)
        # finding the most reachable domain
        max_reached_dom_prob = 0.0
        #
        for n in leaves:
            stag = gpp.node[n]['tag']
            if stag not in accepted_tags:
                continue
            selps = get_dom_trans_prob(tagdomains[stag], domain)
            for dn in tagdomains[stag]:
                if dn not in domainclouds[domainname]:
                    continue
                sp = selps[dn] * gpp.node[n][domainname]['reach_prob_domain']
                if sp > max_reached_dom_prob:
                    max_reached_dom_prob = sp
        reachable_dom_probs.append(max_reached_dom_prob)

        table = get_table_from_domain(domainname, dtype)

        sp = max_reached_dom_prob

        domain_success_probs[domainname] = sp


        if table not in success_probs:
            success_probs_intersect[table] = (1.0-sp)
            success_probs[table] = sp
        else:
            success_probs_intersect[table] *= (1.0-sp)
            success_probs[table] += sp

        for p in gnodes:
            gpp.node[p]['reach_prob'] += gpp.node[p][domainname]['reach_prob_domain']

        h = gpp

    # the prob of finding a table is the union of the prob of finding its domains
    for t, p in success_probs.items():
        success_probs[t] = 1.0-success_probs_intersect[t]
        if success_probs[t] > 1.0:
            print('table %s has sp > 1.0.' % t)
            success_probs[t] = 1.0

    for p in gnodes:
        h.node[p]['reach_prob'] = h.node[p]['reach_prob']/float(len(domains))

    expected_success = sum(list(success_probs.values()))/float(len(success_probs))
    if expected_success == 0:
        print('zero expected_success.')

    return expected_success, h, success_probs, likelihood, domain_success_probs


def get_domain_edge_probs(gd, domain, leaves, gnodes):

    for p in gnodes:
        if p in leaves:
            continue
        ts, sis = get_trans_prob(gd, p, domain)
        for ch, prob in ts.items():
            gd[p][ch][domain['name']] = {'trans_prob_domain': prob}
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



