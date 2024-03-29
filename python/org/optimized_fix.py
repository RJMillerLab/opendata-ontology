import org.optimized_hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import numpy as np
import copy
import datetime


tagdomains = dict()
domains = []
populations = dict()
domainclouds = dict()
# stop exploring after X times of no improvement fixes
termination_condition = 3000
plateau_termination_condition = 200
fix_count = 0
rhcount = 0
apcount = 0

def init(g, doms, tdoms, dclouds):
    global domains, tagdomains, domainclouds, populations
    domains = doms
    tagdomains = tdoms
    domainclouds = dclouds
    h = g.copy()
    return h


def fix_plus(g, doms, tdoms, dclouds, dtype, domaintags):
    global fix_count
    init(g, doms, tdoms, dclouds)
    #orgg.height(g)
    #orgg.branching_factor(g)
    print('started fixing with %d domains.' % len(domains))
    stats = []
    iteration_likelihoods = []
    h = g.copy()
    max_success, gp, max_success_probs, likelihood, max_domain_success_probs = orgh.get_success_prob_likelihood_fuzzy(h, domains, tagdomains, domainclouds, dtype, domaintags)

    best = gp.copy()
    print('starting with success prob: fuzzy %f' % (max_success))

    fixfunctions = [reduce_height, add_parent]

    # termination condition
    pleateau_count = 0
    prev_max_success = max_success
    #
    for i in range(2):
        print(datetime.datetime.now())
        print('iteration %d' % i)
        initial_sp = max_success
        #level_n = list(orgg.get_leaves(gp))
        #print('bottom up')
        #level_n = orgg.level_down(gp, orgg.level_down(gp, [orgg.get_root(gp)]))
        print('top down')
        level_n = set(set(gp.nodes).difference({orgg.get_root(gp)})).difference(set(orgg.level_down(gp, [orgg.get_root(gp)])))
        #li = 0
        while len(level_n) > 1 and fix_count < termination_condition:
            #li += 1
            #if li > 2:
            #    break
            #print('len(level_n): %d nodes: %d edges: %d' % (len(level_n), len(gp.nodes), len(gp.edges)))
            print('len(level_n): %d' % (len(level_n)))
            hf, ll, sps, levelstats, ls, dsps = fix_level_plus(best.copy(), level_n, max_success, max_success_probs, max_domain_success_probs, [fixfunctions[i]], dtype, domaintags)
            stats.extend(list(levelstats))
            iteration_likelihoods.extend(list(ls))
            #print('after fix_level: node %d edge %d success %f' % (len(hf.nodes), len(hf.edges), ll))
            if ll > max_success:
                print('improving after fixing level from %0.7f to %0.7f.' % (max_success, ll))
                max_success = ll
                best = hf.copy()
                max_success_probs = copy.deepcopy(sps)
                max_domain_success_probs = copy.deepcopy(dsps)
            if max_success == prev_max_success:
                pleateau_count += 1
            else:
                pleateau_count = 0
            if pleateau_count > plateau_termination_condition:
                print('plateau after %d fix iterations' % fix_count)
                break
            level_n = orgg.level_up(hf, level_n)
            #level_n = orgg.level_down(hf, level_n)
            #level_n = []
        print('initial success prob: %f  and best success prob: %f' % (initial_sp, max_success))
        #print('after fix_level: node %d edge %d' % (len(best.nodes), len(best.edges)))
        #orgg.height(best)
        #orgg.branching_factor(best)

        gp = best.copy()
        print('Number of fix() iterations: %d' % fix_count)
        print(datetime.datetime.now())
    return best, stats, iteration_likelihoods, max_success_probs, max_domain_success_probs


def fix_level_plus(g, level, success, success_probs, domain_success_probs, fixfunctions, dtype, domaintags):

    stats = []
    iteration_likelihoods = []
    fixes = what_to_fix(g, level)
    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    max_domain_success_probs = copy.deepcopy(domain_success_probs)
    best = g.copy()
    bnodes = best.nodes
    for f in fixes:
        #if f[0] not in best.nodes:
        if f[0] not in bnodes:
            continue
        if len(list(best.predecessors(f[0]))) == 0:
            continue
        if f[1] == 1.0:
            continue
        for ffunc in fixfunctions:
            #print('nodes: %d edges: %d before ffunc' % (len(best.nodes), len(best.edges)))
            start = datetime.datetime.now()
            hp, newsuccess, newsps, its, ls, dsps = ffunc(best.copy(), level, f[0], max_success, max_success_probs, dtype, domaintags, max_domain_success_probs, bnodes)
            print('fix time: %d' % (int((datetime.datetime.now()-start).total_seconds() *1000)))
            print('------------------------')
            #print('nodes: %d edges: %d after ffunc' % (len(hp.nodes), len(hp.edges)))
            if newsuccess < 0.0:
                continue
            if newsuccess > max_success:
                best = hp.copy()
                max_success = newsuccess
                max_success_probs = copy.deepcopy(newsps)
                max_domain_success_probs = copy.deepcopy(dsps)
                bnodes = best.nodes
            #else:
            #    print('opeartor did not help.')
            stats.extend(list(its))
            iteration_likelihoods.extend(list(ls))
    return best, max_success, max_success_probs, stats, iteration_likelihoods, max_domain_success_probs


def add_parent(g, level, n, success, success_probs, dtype, domaintags, domain_success_probs, gnodes):
    global apcount
    print('add_parent')
    #if n not in g.nodes:
    if n not in gnodes:
        return g, -1.0, [], [], [], []
    global fix_count
    fix_count += 1

    iteration_success_probs = []
    iteration_likelihoods = []
    gp = g.copy()
    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    max_domain_success_probs = copy.deepcopy(domain_success_probs)
    best = g.copy()

    leaves = orgg.get_leaves_plus(gp, gnodes)
    choices = (((set(gnodes).difference(set(nx.descendants(gp, n)))).difference(set(gp.predecessors(n)))).difference({n})).difference(leaves)
    schoices = find_second_parent(g, choices)
    print('fix %d' % n)
    num_active_domains, num_prune_domains = 0, 0
    potentials2 = []
    for sp in schoices[:1]:#[:2]:
        p = sp[0]
        h = best.copy()


        if len(list(h.predecessors(p))) > 1:
            print('multiple grand parents')
        update_head1 = nx.lowest_common_ancestor(best, n, p)

        hap, update_head = update_graph_add_parent(best.copy(), p, n)
        if update_head1 != update_head:
            print('update_head gets updated')

        potentials2 = list(nx.descendants(best, update_head))


        new, gl, sps, likelihood, dsps, num_active_domains, num_prune_domains = orgh.get_success_prob_likelihood_partial(hap.copy(), domains, tagdomains, domainclouds, dtype, domaintags, potentials2, update_head, max_success_probs, max_domain_success_probs)
        apcount += 1
        print('after adding parent: prev %f new %f' % (max_success, new))

        iteration_success_probs.append(new)
        iteration_likelihoods.append(likelihood)
        if new > max_success:
            print('connecting to %d improved from %0.7f to %0.7f.' % (p, max_success, new))
            max_success = new
            max_success_probs = copy.deepcopy(sps)
            best = gl.copy()
            max_domain_success_probs = copy.deepcopy(dsps)
            print('fixing %d: adding parent: %d' % (n, p))
            return best, max_success, max_success_probs, [{'active_domains': num_active_domains, 'active_states': len(potentials2), 'prune_domains': num_prune_domains}], iteration_likelihoods, max_domain_success_probs

    return best, max_success, max_success_probs, [{'active_domains': num_active_domains, 'active_states': len(potentials2), 'prune_domains':     num_prune_domains}], iteration_likelihoods, max_domain_success_probs


def what_to_fix(g, nodes):
    m = {n:g.node[n]['reach_prob'] for n in nodes}
    return sorted(m.items(), key=operator.itemgetter(1))


def update_graph_add_parent(g, p, c):

    h = g.copy()
    prev_leaves = orgg.get_leaves(h)
    to_update = []
    ancestors_c = set(nx.ancestors(h,c))
    ancestors_p = set(nx.ancestors(h,p))
    if p in list(ancestors_c):
        to_update = []
    else:
        #if len(set(nx.ancestors(h,p)).difference(set(nx.ancestors(h,c)))) > 0:
        if len(ancestors_p.difference(ancestors_c)) > 0:
            print('something to update')
        #to_update = list((set(nx.ancestors(h,p)).difference(set(nx.ancestors(h,c)))).    union({p}))
        to_update = list((ancestors_p.difference(ancestors_c)).union({p}))
    update_head = nx.lowest_common_ancestor(h, p, c)
    for u in to_update:
        update_head = nx.lowest_common_ancestor(h, update_head, u)


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


def reduce_height(h, level, n, success, success_probs, dtype, domaintags, domain_success_probs, hnodes):
    start = datetime.datetime.now()
    global rhcount
    print('reduce_height')
    if n not in hnodes:
        return h, -1.0, [], [], [], []
    g = h.copy()

    global fix_count
    fix_count += 1

    max_success = success
    max_success_probs = copy.deepcopy(success_probs)
    max_domain_success_probs = copy.deepcopy(domain_success_probs)
    best = h.copy()
    iteration_success_probs = []
    iteration_likelihoods = []

    parents = list(g.predecessors(n))
    # choose the least reachable parent
    pfixes = what_to_fix(g, parents)
    pf = pfixes[0]
    grandparents = list(g.predecessors(pf[0]))
    if len(grandparents) == 0:
        return g, -1.0, [], [], [], []
    # mix the siblings from the least reachable grand parent
    gpfixes = what_to_fix(g, grandparents)
    gpf = gpfixes[0]
    hp = merge_siblings_and_replace_parent(g, gpf[0], hnodes)

    potentials = list(set(nx.descendants(hp,gpf[0])))

    print('changing org time: %d' % ((datetime.datetime.now()-start).total_seconds()*1000))

    new, gl, sps, likelihood, dsps, num_active_domains, num_prune_domains = orgh.get_success_prob_likelihood_partial(hp.copy(), domains, tagdomains, domainclouds, dtype, domaintags, potentials, gpf[0], success_probs, domain_success_probs)
    rhcount += 1

    iteration_success_probs.append(new)
    iteration_likelihoods.append(likelihood)
    if new > max_success:
        print('reducing height improved from %0.7f to %0.7f.' % (max_success, new))
        max_success = new
        max_success_probs = copy.deepcopy(sps)
        best = gl.copy()
        max_domain_success_probs = copy.deepcopy(dsps)

    print('after reduction: prev %0.7f new %0.7f' % (max_success, new))
    return best, max_success, max_success_probs, [{'active_domains': num_active_domains, 'active_states': len(potentials), 'prune_domains': num_prune_domains}], iteration_likelihoods, max_domain_success_probs


def merge_siblings_and_replace_parent(g, p, gnodes):
    h = g.copy()
    leaves = orgg.get_leaves_plus(g, gnodes)
    sibs = list(h.successors(p))
    for s in sibs:
        if s in leaves:
            continue
        for n in list(h.successors(s)):
            if p not in nx.descendants(h, n):
                h.add_edge(p, n)
        h.remove_edge(p, s)
        if len(list(h.predecessors(s))) == 0:
            for n in list(h.successors(s)):
                h.remove_edge(s, n)
            h.remove_node(s)
        else:
            print('not removing the node')

    return h


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






