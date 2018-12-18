import org.optimized_hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import math
import numpy as np
import copy
import datetime


tagdomains = dict()
domains = []
populations = dict()
domainclouds = dict()
rhcount = 0
apcount = 0


def init(g, doms, tdoms, dclouds):
    global domains, tagdomains, domainclouds, populations
    domains = doms
    tagdomains = tdoms
    domainclouds = dclouds
    h = g.copy()
    for n in h.nodes:
        populations[n] = copy.deepcopy(h.node[n]['population'])
        h.node[n]['population'] = []
    return h



def fix_plus(g, doms, tdoms, dclouds, dtype, domaintags):
    init(g, doms, tdoms, dclouds)
    #orgg.height(g)
    #orgg.branching_factor(g)
    print('started fixing with %d domains.' % len(domains))
    iteration_success_probs = []
    iteration_likelihoods = []
    h = g.copy()

    max_success, gp, max_success_probs, likelihood, max_domain_success_probs = orgh.get_success_prob_likelihood_fuzzy(h, domains, tagdomains, domainclouds, dtype, domaintags)

    org = orgg.graph_to_org(gp)

    best_org = copy.deepcopy(org)
    best = gp.copy()
    print('starting with success prob: fuzzy %f' % (max_success))

    fixfunctions = [reduce_height, add_parent]

    for i in range(2):
        print(datetime.datetime.now())
        print('iteration %d' % i)
        initial_sp = max_success
        #level_n = list(orgg.get_leaves(gp))
        #print('bottom up')
        level_n = orgg.level_down(gp, orgg.level_down(gp, [orgg.get_root(gp)]))
        print('top down')
        #level_n = set(set(gp.nodes).difference({orgg.get_root(gp)})).difference(set(orgg.level_down(gp, [orgg.get_root(gp)])))
        first = True
        while len(level_n) > 1 and first:
            first = False
            print('len(level_n): %d nodes: %d edges: %d' % (len(level_n), len(gp.nodes), len(gp.edges)))
            hf, ll, sps, its, ls, dsps, norg = fix_level_plus(best.copy(), level_n, max_success, max_success_probs, max_domain_success_probs, [fixfunctions[i]], dtype, domaintags, best_org)
            iteration_success_probs.extend(list(its))
            iteration_likelihoods.extend(list(ls))
            #print('after fix_level: node %d edge %d success %f' % (len(hf.nodes), len(hf.edges), ll))
            if ll > max_success:
                print('improving after fixing level from %f to %f.' % (max_success, ll))
                max_success = ll
                best = hf.copy()
                best_org = copy.deepcopy(norg)
                max_success_probs = copy.deepcopy(sps)
                max_domain_success_probs = copy.deepcopy(dsps)
            #level_n = orgg.level_up(hf, level_n)
            level_n = orgg.level_down(hf, level_n)
            #level_n = []
        print('initial success prob: %f  and best success prob: %f' % (initial_sp, max_success))
        print('after fix_level: node %d edge %d' % (len(best.nodes), len(best.edges)))
        #orgg.height(best)
        #orgg.branching_factor(best)

        gp = best.copy()
        print(datetime.datetime.now())
    print('rhcount : %d and apcount: %d' % (rhcount, apcount))
    return best, iteration_success_probs, iteration_likelihoods, max_success_probs, max_domain_success_probs


def fix_level_plus(g, level, success, success_probs, domain_success_probs, fixfunctions, dtype, domaintags, org):

    iteration_success_probs = []
    iteration_likelihoods = []
    fixes = what_to_fix(g, level)
    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    max_domain_success_probs = copy.deepcopy(domain_success_probs)
    best = g.copy()
    best_org = copy.deepcopy(org)
    for f in fixes:
        # the node was removed by reduce_height
        if f[0] not in best.nodes:
            continue
        if len(list(best.predecessors(f[0]))) == 0:
            continue
        if f[1] == 1.0:
            continue
        for ffunc in fixfunctions:
            print('nodes: %d edges: %d before ffunc' % (len(best.nodes), len(best.edges)))
            hp, newsuccess, newsps, its, ls, dsps, norg = ffunc(best.copy(), level, f[0], max_success, max_success_probs, dtype, domaintags, max_domain_success_probs, copy.deepcopy(best_org))
            print('nodes: %d edges: %d after ffunc' % (len(hp.nodes), len(hp.edges)))
            if newsuccess > max_success:
                best = hp.copy()
                best_org = copy.deepcopy(norg)
                max_success = newsuccess
                max_success_probs = copy.deepcopy(newsps)
                max_domain_success_probs = copy.deepcopy(dsps)
            iteration_success_probs.extend(list(its))
            iteration_likelihoods.extend(list(ls))
    return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods, max_domain_success_probs, best_org


def add_parent(g, level, n, success, success_probs, dtype, domaintags, domain_success_probs, org):
    global apcount
    print('add_parent')
    if n not in org['nodes']:
        return g, -1.0, [], [], [], []

    iteration_success_probs = []
    iteration_likelihoods = []
    gp = g.copy()
    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    max_domain_success_probs = copy.deepcopy(domain_success_probs)
    best = g.copy()
    best_org = copy.deepcopy(org)

    #leaves = orgg.get_leaves(gp)
    leaves = org['leaves']
    choices = (((set(gp.nodes).difference(set(nx.descendants(gp, n)))).difference(set(gp.predecessors(n)))).difference({n})).difference(leaves)
    #schoices = sort_nodes_sim(g, n, choices)
    schoices = find_second_parent(g, choices, org)
    print('fix %d' % n)
    for sp in schoices[:2]:
        p = sp[0]
        h = best.copy()

        if len(list(h.predecessors(p))) > 1:
            print('multiple grand parents')

        hap, update_head, norg = update_graph_add_parent(h, p, n, copy.deepcopy(best_org))

        # update best or modified one?
        potentials2 = list(nx.descendants(best, update_head))

        new, gl, sps, likelihood, dsps, nnorg = orgh.get_success_prob_likelihood_partial(hap.copy(), domains, tagdomains, domainclouds, dtype, domaintags, potentials2, update_head, max_success_probs, max_domain_success_probs, norg)
        print('after eval: nodes: %d edge: %d' % (len(gl.nodes), len(gl.edges)))
        apcount += 1
        print('after add_parent: prev %f new %f' % (max_success, new))

        iteration_success_probs.append(new)
        iteration_likelihoods.append(likelihood)
        if new > max_success:
            print('connecting to %d improved from %f to %f.' % (p, max_success, new))
            max_success = new
            max_success_probs = copy.deepcopy(sps)
            best = gl.copy()
            best_org = copy.deepcopy(norg)
            max_domain_success_probs = copy.deepcopy(dsps)
            print('fixing %d: adding parent: %d' % (n, p))
            return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods, max_domain_success_probs, best_org

        #iteration_success_probs.append(max_success)
    return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods, max_domain_success_probs


def what_to_fix(g, nodes, org):
    m = {}
    for n in nodes:
        m[n] = org['nodes'][n]['reach_prob']
    return sorted(m.items(), key=operator.itemgetter(1))


def update_graph_add_parent(g, p, c, org):

    h = g.copy()
    prev_leaves = orgg.get_leaves(h)
    to_update = []
    if p in list(nx.ancestors(h, c)):
        to_update = []
    else:
        if len(set(nx.ancestors(h,p)).difference(set(nx.ancestors(h,c)))) > 0:
            print('something to update')
        to_update = list((set(nx.ancestors(h,p)).difference(set(nx.ancestors(h,c)))).union({p}))
    update_head = nx.lowest_common_ancestor(h, p, c)
    for u in to_update:
        update_head = nx.lowest_common_ancestor(h, update_head, u)


    print('to_update: %d' % len(to_update))
    h.add_edge(p, c)
    leaves = orgg.get_leaves(h)
    for n in to_update:
        pops = []
        to_add = []
        if n in prev_leaves:
            to_add.append(n)
        to_add.extend(list((set(nx.descendants(h,n)).intersection(set(leaves)))))
        to_add = list(set(to_add))
        for a in to_add:
            pops.append(h.node[a]['rep'])
            if h.node[a]['tag'] not in h.node[n]['tags']:
                h.node[n]['tags'].append(h.node[a]['tag'])
        if len(pops) > 0:
            h.node[n]['rep'] = list(np.mean(np.array(pops), axis=0))
    orgh.update_node_dom_sims(h, domains, to_update)
    return h, update_head


def reduce_height(h, level, n, success, success_probs, dtype, domaintags, domain_success_probs, org):
    global rhcount
    print('reduce_height')
    if n not in org['nodes']:
        print('node has been removed')
        return h, -1.0, [], [], [], []
    g = h.copy()

    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    max_domain_success_probs = copy.deepcopy(domain_success_probs)
    best = h.copy()
    best_org = copy.deepcopy(org)
    iteration_success_probs = []
    iteration_likelihoods = []

    parents = list(g.predecessors(n))
    # choose the least reachable parent
    pfixes = what_to_fix(g, parents, org)
    pf = pfixes[0]
    grandparents = list(g.predecessors(pf[0]))
    if len(grandparents) == 0:
        print('no grandparents')
        return g, -1.0, [], [], [], []
    # mix the siblings from the least reachable grand parent
    gpfixes = what_to_fix(g, grandparents, org)
    gpf = gpfixes[0]
    hp, norg = merge_siblings_and_replace_parent(g, gpf[0], org)

    potentials = list(set(nx.descendants(hp,gpf[0])))

    new, gl, sps, likelihood, dsps, nnorg = orgh.get_success_prob_likelihood_partial(hp.copy(), domains, tagdomains, domainclouds, dtype, domaintags, potentials, gpf[0], success_probs, domain_success_probs, norg)
    rhcount += 1

    iteration_success_probs.append(new)
    iteration_likelihoods.append(likelihood)
    if new > max_success:
        print('reducing height improved from %f to %f.' % (max_success, new))
        max_success = new
        max_success_probs = copy.deepcopy(sps)
        best = gl.copy()
        best_org = copy.deepcopy(nnorg)
        max_domain_success_probs = copy.deepcopy(dsps)
    print('after reduction: prev %f new %f' % (max_success, new))
    return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods, max_domain_success_probs, best_org


def merge_siblings_and_replace_parent(g, p, org):
    h = g.copy()
    #leaves = orgg.get_leaves(g)
    leaves = org['leaves']
    sibs = list(h.successors(p))
    for s in sibs:
        if s in leaves:
            continue
        for n in list(h.successors(s)):
            if p not in nx.descendants(h, n):
                h.add_edge(p, n)
                org['edges'][p][n] = dict()
        h.remove_edge(p, s)
        del org['edges'][p][s]
        if len(list(h.predecessors(s))) == 0:
            for n in list(h.successors(s)):
                h.remove_edge(s, n)
                del org['edges'][s][n]
            h.remove_node(s)
            del org['nodes'][s]
        else:
            print('not removing the node')

    return h, org


def update_width(g, c):
    success = 0.0
    leaves = orgg.get_leaves(g)
    to_update = list(set(nx.descendants(g,c)).intersection(set(leaves)))
    for domain in domains:
        if domain in to_update:
            success += orgh.local_log_success(g, domain)
        else:
            for n in g.nodes:
                success += math.log(g.node[n][domain['name']]['reach_prob_domain'])

    for p in g.nodes:
        g.node[p]['reach_prob'] = 0.0
        for domain in domains:
            g.node[p]['reach_prob'] += g.node[p][domain['name']]['reach_prob_domain']
    for p in to_update:
        g.node[p]['reach_prob'] = g.node[p]['reach_prob']/float(len(domains))

    return success, g


def find_another_parent(g, n, level):
    cands = dict()
    #level_up = orgg.level_up(g, level)
    level_up = (((set(g.nodes).difference(set(nx.descendants(g, n)))).difference(set(g.predecessors(n)))).difference({n})).difference(orgg.get_leaves(g))

    if len(level_up) == 0:
        return -1
    for c in level_up:
        cands[c] = orgh.get_transition_sim(g.node[n]['rep'], g.node[c]['rep'])
    scands = sorted(cands.items(), key=operator.itemgetter(1), reverse=True)
    cand = scands[0][0]
    i = 1
    while i < len(scands) and (cand in list(g.predecessors(n)) or cand in list(nx.descendants(g,n))):
        cand = scands[i][0]
        i += 1
    return cand


def sort_nodes_sim(g, c, nodes):
    sims = dict()
    for n in nodes:
        sims[n] = orgh.get_transition_sim(g.node[n]['rep'], g.node[c]['rep'])
    ssims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True)
    return ssims


def find_second_parent(g, nodes, org):
    m = {}
    for n in nodes:
        m[n] = org['nodes'][n]['reach_prob']
    return sorted(m.items(), key=operator.itemgetter(1), reverse=True)






