import org.hierarchy as orgh
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
    orgg.height(g)
    orgg.branching_factor(g)
    print('started fixing with %d domains.' % len(domains))
    iteration_success_probs = []
    iteration_likelihoods = []
    h = g
    #max_success, gp, max_success_probs, likelihood = orgh.get_success_prob_likelihood(h, domains)
    max_success, gp, max_success_probs, likelihood = orgh.get_success_prob_likelihood_fuzzy(h, domains, tagdomains, domainclouds, dtype, domaintags)
    #initial_success_probs = copy.deepcopy(max_success_probs)
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
        while len(level_n) > 1:
            print('len(level_n): %d nodes: %d edges: %d' % (len(level_n), len(gp.nodes), len(gp.edges)))
            hf, ll, sps, its, ls = fix_level_plus(best.copy(), level_n, max_success, max_success_probs, [fixfunctions[i]], dtype, domaintags)
            iteration_success_probs.extend(list(its))
            iteration_likelihoods.extend(list(ls))
            #print('after fix_level: node %d edge %d success %f' % (len(hf.nodes), len(hf.edges), ll))
            if ll > max_success:
                print('improving after fixing level from %f to %f.' % (max_success, ll))
                max_success = ll
                best = hf.copy()
                max_success_probs = copy.deepcopy(sps)
            #level_n = orgg.level_up(hf, level_n)
            level_n = orgg.level_down(hf, level_n)
            #level_n = []
        print('initial success prob: %f  and best success prob: %f' % (initial_sp, max_success))
        #print('improvement in success probs: %f' % orgh.get_improvement(initial_success_probs, max_success_probs))
        #print('after fix_level: node %d edge %d' % (len(best.nodes), len(best.edges)))
        orgg.height(best)
        orgg.branching_factor(best)

        gp = best.copy()
        print(datetime.datetime.now())
    return best, iteration_success_probs, iteration_likelihoods


def fix_level_plus(g, level, success, success_probs, fixfunctions, dtype, domaintags):
    iteration_success_probs = []
    iteration_likelihoods = []
    fixes = what_to_fix(g, level)
    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    best = g.copy()
    for f in fixes:
        if f[0] not in best.nodes:
            continue
        if len(list(best.predecessors(f[0]))) == 0:
            continue
        if f[1] == 1.0:
            continue
        for ffunc in fixfunctions:
            hp, newsuccess, newsps, its, ls = ffunc(best.copy(), level, f[0], success, max_success_probs, dtype, domaintags)
            if newsuccess < 0.0:
                continue
            if newsuccess > max_success:
                best = hp.copy()
                max_success = newsuccess
                max_success_probs = copy.deepcopy(newsps)
                success = newsuccess
            #else:
            #    print('opeartor did not help.')
            iteration_success_probs.extend(list(its))
            iteration_likelihoods.extend(list(ls))
    return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods


def add_parent(g, level, n, success, success_probs, dtype, domaintags):
    print('add_parent')
    if n not in g.nodes:
        return g, -1.0, [], [], []

    iteration_success_probs = []
    iteration_likelihoods = []
    gp = g
    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    best = g.copy()
    #choices = list((set(orgg.level_up(gp, level)).difference(set(nx.descendants(gp, n)))).difference(gp.predecessors(n)))
    choices = ((set(gp.nodes).difference(set(nx.descendants(gp, n)))).difference(set(gp.predecessors(n)))).difference({n})
    #schoices = sort_nodes_sim(g, n, choices)
    schoices = find_second_parent(g, choices)
    print('fix %d' % n)
    for sp in schoices[:2]:
        p = sp[0]
        h = gp.copy()
        ans1 = list(nx.ancestors(h, p))
        ans1.append(p)
        ans2 = list(nx.ancestors(h, n))
        ans2.append(n)
        ans1 = set(ans1)
        ans2 = set(ans2)
        #updates = []
        if p in list(nx.ancestors(h, n)):
            ans = ans2.difference(ans1)
            ans = list(ans.union({n}))
            #updates = [n]
        else:
            ans = ans1.difference(ans2)
            ans = list(ans.union({n, p}))
            #updates = list(ans)
        potentials = list(ans)
        hap = update_graph_add_parent(h, p, n)
        for a in ans:
            for d in list(hap.predecessors(a)):
                t = set(potentials+list(nx.descendants(hap, d)))
                potentials = list(t)
        #new, gl, sps, likelihood  = orgh.recompute_success_prob_likelihood(hap.copy(), domains, potentials, tagdomains, True, success_probs)
        #new, gl, sps, likelihood  = orgh.recompute_success_prob_likelihood_fuzzy(hap.copy(), domains, potentials, tagdomains, True, success_probs, domainclouds)

        #new, gl, sps, likelihood = orgh.get_success_prob_likelihood(hap.copy(), domains)
        new, gl, sps, likelihood = orgh.get_success_prob_likelihood_fuzzy(hap.copy(), domains, tagdomains, domainclouds, dtype, domaintags)

        iteration_success_probs.append(new)
        iteration_likelihoods.append(likelihood)
        if new > max_success:
            print('connecting to %d improved from %f to %f.' % (p, max_success, new))
            max_success = new
            max_success_probs = copy.deepcopy(sps)
            best = gl.copy()
            print('fixing %d: adding parent: %d' % (n, p))
            return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods

        #iteration_success_probs.append(max_success)
    return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods


def what_to_fix(g, nodes):
    m = {}
    for n in nodes:
        m[n] = g.node[n]['reach_prob']
    return sorted(m.items(), key=operator.itemgetter(1))


def update_graph_add_parent(g, p, c):
    #h = g.copy()
    h = g
    leaves = orgg.get_leaves(h)
    to_update = []
    if p in list(nx.ancestors(g, c)):
        to_update = []
    else:
        to_update = list((set(nx.ancestors(h,p)).difference(nx.ancestors(h,c))).union({p}))
    h.add_edge(p, c)
    for n in to_update:
        #h.node[n]['population'] = []
        pops = []
        to_add = list((set(nx.descendants(h,n)).intersection(set(leaves))))
        for a in to_add:
            #h.node[n]['population'].append(h.node[a]['rep'])
            pops.append(h.node[a]['rep'])
            if h.node[a]['tag'] not in h.node[n]['tags']:
                h.node[n]['tags'].append(h.node[a]['tag'])
        #h.node[n]['rep'] = list(np.mean(np.array(h.node[n]['population']), axis=0))
        h.node[n]['rep'] = list(np.mean(np.array(pops), axis=0))
    orgh.update_node_dom_sims(h, domains, to_update)
    return h


def reduce_height(h, level, n, success, success_probs, dtype, domaintags):
    print('reduce_height')
    if n not in h.nodes:
        return h, -1.0, [], [], []
    g = h

    #print('fix node %d' % n)
    #orgg.gprint(g)

    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    best = g.copy()
    iteration_success_probs = []
    iteration_likelihoods = []

    parents = list(g.predecessors(n))
    # choose the least reachable parent
    pfixes = what_to_fix(g, parents)
    pf = pfixes[0]
    #print('fixing parent %d with %f' % (pf[0], pf[1]))
    grandparents = list(g.predecessors(pf[0]))
    if len(grandparents) == 0:
        #print('got to the root')
        return g, -1.0, [], [], []
    # mix the siblings from the least reachable grand parent
    gpfixes = what_to_fix(g, grandparents)
    gpf = gpfixes[0]
    hp = merge_siblings_and_replace_parent(h, gpf[0])

    #new, gl, sps, likelihood = orgh.get_success_prob_likelihood(hp.copy(), domains)
    new, gl, sps, likelihood = orgh.get_success_prob_likelihood_fuzzy(hp.copy(), domains, tagdomains, domainclouds, dtype, domaintags)

    iteration_success_probs.append(new)
    iteration_likelihoods.append(likelihood)
    if new > max_success:
        print('reducing height improved from %f to %f.' % (max_success, new))
        max_success = new
        max_success_probs = copy.deepcopy(sps)
        best = gl.copy()
    print('after reduction: prev %f new %f' % (max_success, new))
    return best, max_success, max_success_probs, iteration_success_probs, iteration_likelihoods


def merge_siblings_and_replace_parent(h, p):
    #print('merging the children of %d' % p)
    sibs = list(h.successors(p))
    for s in sibs:
        if h.out_degree(s)==0:
            continue
        for n in list(h.successors(s)):
            h.add_edge(p, n)
        h.remove_edge(p, s)
        if len(list(h.predecessors(s))) == 0:
            h.remove_node(s)
    return h


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


def find_second_parent(g, nodes):
    m = {}
    for n in nodes:
        m[n] = g.node[n]['reach_prob']
    return sorted(m.items(), key=operator.itemgetter(1), reverse=True)






