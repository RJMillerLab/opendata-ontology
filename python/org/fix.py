import org.hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import math
import numpy as np


sims = dict()

def fix(g, domains):
    print('len domains: %d' % len(domains))
    print('started fixing')
    init(g, domains)
    h, hf = g.copy(), g.copy()
    #max_likelihood, gp = orgh.log_likelihood(h, domains)
    max_likelihood, gp = orgh.success_prob(h, domains)
    initial_score = max_likelihood
    best = gp
    for fix_round in range(1,3):
        level_n = list(orgg.get_leaves(gp))
        curr_ll = max_likelihood
        while len(level_n) > 1:
            print('len(level_n): %d nodes: %d edges: %d' % (len(level_n), len(gp.nodes), len(gp.edges)))
            if fix_round == 1:
                print('fixing by changing parent.')
                hf, ll = fix_level(gp, level_n, domains, curr_ll, fix_node_change_parent)
                curr_ll = ll
            else:
                print('fixing by adding parent.')
                hf, ll = fix_level(gp, level_n, domains, curr_ll, fix_node_add_parent)
                curr_ll = ll
            print('after fix_level: node %d edge %d likelihood %f' % (len(hf.nodes), len(hf.edges), curr_ll))
            if curr_ll > max_likelihood:
                print('improving after fixing level')
                max_likelihood = curr_ll
                best = hf
                #orgh.evaluate(hf, domains)
            level_n = orgg.level_up(hf, level_n)
            print('done with this step.')
        orgh.evaluate(best, domains)
        gp = best.copy()
    print('initial score: %f', initial_score)
    print('best score: %f' % max_likelihood)
    return best


def fix_level(g, level, domains, likelihood, fixfunction):
    h = g.copy()
    fixes = what_to_fix(h, level)
    max_likelihood = likelihood
    best = g.copy()
    for f in fixes:#[:min(1,len(fixes))]:
        if f[0] not in h.nodes:
            continue
        if len(list(h.predecessors(f[0]))) == 0:
            continue
        if f[1] == 1.0:
            continue
        hp, likelihood = fixfunction(h, level, f[0], domains, likelihood)
        if likelihood > max_likelihood:
            print('improving level.')
            max_likelihood = likelihood
            best = hp.copy()
        h = hp.copy()
    return best, max_likelihood


def change_parent(g, n, level, domains):
    parents = list(g.predecessors(n))
    print('finding another parent for %d' % n)
    newparent = find_another_parent(g, n, level)
    if newparent == -1:
        return g, -1.0
    print('newparent: %d' % newparent)
    pfixes = what_to_fix(g, parents)
    oldparent = pfixes[0][0]
    # nodes to be updated
    ans1 = list(nx.ancestors(g, oldparent))
    ans1.append(oldparent)
    ans2 = list(nx.ancestors(g, newparent))
    ans2.append(newparent)
    ans1 = set(ans1)
    ans2 = set(ans2)
    ans = ans1.union(ans2).difference(ans1.intersection(ans2))
    ans = list(ans.union({n}))
    potentials = ans
    for a in ans:
        for d in list(g.predecessors(a)):
            potentials = list(set(potentials+list(nx.descendants(g, d))))
    h = update_graph_change_parent(g.copy(), oldparent, n, newparent)
    new, np = orgh.recompute_success_prob(h, domains, potentials)
    #new, np = orgh.log_likelihood(i, domains)
    #new, np = orgh.success_prob(h, domains)
    print('potentials: %d' % len(potentials))
    return np, new


def add_parent(g, level, n, domains, likelihood):
    print('before add_parent: node %d edge %d' % (len(g.nodes), len(g.edges)))
    if n not in g.nodes:
        return g, -1, -1
    gp = g.copy()
    max_likelihood = likelihood
    best = gp
    choices = list((set(orgg.level_up(gp, level)).difference(set(nx.descendants(gp, n)))).difference(gp.predecessors(n)))
    print('choices: %d' % len(choices))
    bestparent = -1
    schoices = sort_nodes_sim(g, n, choices)
    for sp in schoices:
        p = sp[0]
        h = gp.copy()
        ans = list(set(nx.ancestors(h, p)).difference(nx.ancestors(h, n)))
        ans.append(p)
        ans.append(n)
        potentials = ans
        for a in ans:
            for d in list(h.predecessors(a)):
                potentials = list(set(potentials+list(nx.descendants(h, d))))
        h = update_graph_add_parent(h, p, n)
        #new, gl = update_likelihood_width(h, p, domains)
        #new, gl = orgh.log_likelihood(h, domains)
        #new, gl = orgh.success_prob(h, domains)
        new, gl = orgh.recompute_success_prob(h, domains, potentials)
        print('potential: %d' % len(potentials))
        if new > max_likelihood:
            print('connecting to %d improved' % p)
            max_likelihood = new
            bestparent = p
            best = gl
            gp = h
            return best, bestparent, max_likelihood
    #print('after add_parent: node %d edge %d' % (len(best.nodes), len(best.edges)))
    return best, bestparent, max_likelihood


# incrementally updating log likelihood when adding an edge from p to c.
def update_likelihood_height(g, p, topo_order, domains):
    for domain in domains:
        tps = get_trans_prob(g, p, domain)
        for s in list(g.successors(p)):
            if domain['name'] not in g[p][s]:
                g[p][s][domain['name']] = dict()
            if s not in tps:
                print('y')
            g[p][s][domain['name']]['trans_prob_domain'] = tps[s]

    to_update = list(nx.descendants(g, p))

    for domain in domains:
        for s in list(g.successors(p)):
            g.node[s][domain['name']]['reach_prob_domain'] = g.node[p][domain['name']]['reach_prob_domain'] * g[p][s][domain['name']]['trans_prob_domain']
        for s in to_update:
            g.node[s][domain['name']]['reach_prob_domain'] = 0.0
        for u in topo_order:
            for ch in list(g.successors(u)):
                if ch not in to_update:
                    continue
                g.node[ch][domain['name']]['reach_prob_domain'] += g.node[u][domain['name']]['reach_prob_domain']*g[u][ch][domain['name']]['trans_prob_domain']
                if g.node[ch][domain['name']]['reach_prob_domain']>1.0:
                    print('%s > 1.0' % domain['name'])

    for p in to_update:
        g.node[p]['state_loglikelihood'] = 0.0
        g.node[p]['reach_prob'] = 0.0
        for domain in domains:
            g.node[p]['reach_prob'] += g.node[p][domain['name']]['reach_prob_domain']
            g.node[p]['state_loglikelihood'] += math.log(g.node[p][domain['name']]['reach_prob_domain'], 10)
    for p in to_update:
        g.node[p]['reach_prob'] = g.node[p]['reach_prob']/float(len(domains))
    log_likelihood = 0.0

    for n in g.nodes:
        log_likelihood += g.node[p]['state_loglikelihood']

    return log_likelihood, g


def what_to_fix(g, nodes):
    m = {}
    for n in nodes:
        m[n] = g.node[n]['reach_prob']
    return sorted(m.items(), key=operator.itemgetter(1))


def init(g, domains):
    global sims
    #h = calculate_states(g, domains)
    sims = get_trans_matrix(g, domains)
    #return h


# calculate local log likelihood of the state and reachability probability given a domain
def calculate_states(g, domains):
    h = g.copy()
    for domain in domains:
        gp = orgh.get_domain_edge_probs(h, domain)
        gpp = orgh.get_domain_node_probs(gp, domain)
        for n in gpp.nodes:
            if domain['name'] not in gpp.node[n]:
                gpp.node[n][domain['name']] = dict()
            if 'state_loglikelihood' not in gpp.node[n]:
                gpp.node[n][domain['name']]['reach_prob_domain'] = gpp.node[n][domain['name']]['reach_prob_domain']
                gpp.node[n]['state_loglikelihood'] = math.log(gpp.node[n][domain['name']]['reach_prob_domain'], 10)
                gpp.node[n]['reach_prob'] = gpp.node[n][domain['name']]['reach_prob_domain']
            else:
                gpp.node[n][domain['name']]['reach_prob_domain'] = gpp.node[n][domain['name']]['reach_prob_domain']
                gpp.node[n][domain['name']]['state_loglikelihood'] += math.log(gpp.node[n][domain['name']]['reach_prob_domain'], 10)
                gpp.node[n][domain['name']]['reach_prob'] += gpp.node[n][domain['name']]['reach_prob_domain']
        for e in gpp.edges:
            p, ch = e[0], e[1]
            if domain['name'] not in gpp[p][ch]:
                gpp[p][ch][domain['name']] = dict()
            gpp[p][ch][domain['name']]['trans_prob_domain'] = gpp[p][domain['name']][ch]['trans_prob_domain']
            gpp[p][ch][domain['name']]['trans_sim_domain'] = gpp[p][domain['name']][ch]['trans_sim_domain']
        h = gpp

    for n in h.nodes:
        h.node[n]['reach_prob'] = h.node[n]['reach_prob']/float(len(domains))

    return h


def get_trans_matrix(g, domains):
    sims = dict()
    for n in g.nodes:
        sims[n] = dict()
        for domain in domains:
            sims[n][domain['name']] = orgh.get_transition_sim(g.node[n]['rep'], domain['mean'])
    return sims


def get_trans_prob(g, p, domain):
    d = 0.0
    tps = dict()
    tsl = []
    ts = dict()
    sps = list(g.successors(p))
    for s in sps:
        if s not in sims:
            print('shout')
        tsl.append(sims[s][domain['name']])
        ts[s] = sims[s][domain['name']]
    maxs = max(tsl)
    for s in sps:
        tps[s] = math.exp(ts[s]-maxs)
        d += tps[s]
    for s in sps:
        tps[s] = tps[s]/d
    return tps


def update_graph_plus(g, p, c):
    h = g.copy()
    leaves = orgg.get_leaves(h)
    to_add = list((set(nx.descendants(h, c)).intersection(set(leaves))).difference(set(nx.descendants(h,p)).intersection(set(leaves))))
    h.add_edge(p, c)
    # update population matrix
    for n in to_add:
        h.node[p]['population'].append(h.node[n]['rep'])
    # update rep vector
    h.node[p]['rep'] = list(np.mean(np.array(h.node[p]['population']), axis=0))
    return h


def update_graph_add_parent(g, p, c):
    h = g.copy()
    leaves = orgg.get_leaves(h)
    to_update = list(set(nx.ancestors(h,p)).difference(nx.ancestors(h,c)))
    to_update.append(p)
    h.add_edge(p, c)
    for n in to_update:
        h.node[n]['population'] = []
        # update population matrix
        to_add = list((set(nx.descendants(h,n)).intersection(set(leaves))))
        for a in to_add:
            h.node[n]['population'].append(h.node[a]['rep'])
        # update rep vector
        h.node[n]['rep'] = list(np.mean(np.array(h.node[n]['population']), axis=0))
    return h


def update_graph_change_parent(g, p, c, newp):
    h = g.copy()
    leaves = orgg.get_leaves(h)
    # removing the old parent
    to_remove = list(nx.ancestors(h,p))
    to_remove.append(p)
    to_add = list(nx.ancestors(h,newp))
    to_add.append(newp)
    to_update = list((set(to_remove).union(set(to_add))).difference(set(to_remove).intersection(set(to_add))))
    h.remove_edge(p, c)
    h.add_edge(newp, c)
    for n in to_update:
        vs = list((set(nx.descendants(h,n)).intersection(set(leaves))))
        for a in vs:
            h.node[n]['population'].append(h.node[a]['rep'])
        h.node[n]['rep'] = list(np.mean(np.array(h.node[n]['population']), axis=0))
    return h


def fix_node_add_parent(g, level, n, domains, likelihood):
    max_likelihood = likelihood
    best = g.copy()
    print('fixing n: %d' % n)
    np, bestparent, new = add_parent(g.copy(), level, n, domains, max_likelihood)
    if bestparent == -1:
        return best, max_likelihood
    #new, np = orgh.log_likelihood(h, domains)
    #new, np = orgh.success_prob(h, domains)
    ##new, np = update_likelihood_width(h, bestparent, domains)
    print('new score after the attempt to add width: %f' % new)
    if new > max_likelihood:
        print('improved by adding parent')
        max_likelihood = new
        best = np.copy()
    else:
        print('adding parent did not help.')
    print('done add_parent: node %d edge %d' % (len(np.nodes), len(np.edges)))
    return best, max_likelihood


def fix_node_change_parent(g, level, n, domains, likelihood):
    max_likelihood = likelihood
    best = g.copy()
    np, new = change_parent(g, n, level, domains)
    if new == -1.0:
        print('did not change parent.')
        return g, likelihood
    print('new score after the attempt to change the parent: %f' % new)
    if new > max_likelihood:
        print('improved by add width')
        max_likelihood = new
        best = np.copy()
    else:
        print('changing parent did not help.')
    #print('done change_parent: node %d edge %d' % (len(i.nodes), len(i.edges)))
    return best, max_likelihood


def fix_node_agg(g, level, n, domains, likelihood):
    max_likelihood = likelihood
    gl = g.copy()
    print('starting with node %d and score %f' % (n, max_likelihood))
    best = gl.copy()
    h, updated_gp = reduce_height(gl, n)
    if updated_gp == -1:
        return best, max_likelihood
    topo_order = list(nx.topological_sort(h))
    new, lp = update_likelihood_height(h, updated_gp, topo_order, domains)
    print('new score after the attempt to reduce height: %f' % new)
    if new > max_likelihood:
        print('improved by reducing height')
        max_likelihood = new
        best = lp.copy()
    else:
        print('reducing height did not help.')
    return best, max_likelihood



def fix_node_agg_plus(g, level, n, domains, likelihood):
    max_likelihood = likelihood
    gl = g.copy()
    print('starting with node %d and score %f' % (n, max_likelihood))
    best = gl.copy()
    h, updated_gp = reduce_height(gl, n)
    topo_order = list(nx.topological_sort(h))
    new, lp = update_likelihood_height(h, updated_gp, topo_order, domains)
    #new = orgh.log_likelihood(h, domains)
    print('new score after the attempt to reduce height: %f' % new)
    if new > max_likelihood:
        print('improved by reducing height')
        max_likelihood = new
        best = lp.copy()
    else:
        print('reducing height did not help.')
    #print('reduction done: node %d edge %d' % (len(h.nodes), len(h.edges)))
    hp = best.copy()
    h, bestparent, new = add_parent(hp, level, n, domains, max_likelihood)
    if bestparent == -1:
        return best, max_likelihood
    new, np = update_likelihood_width(h, bestparent, domains)
    print('new score after the attempt to add width: %f' % new)
    if new > max_likelihood:
        print('improved by add width')
        max_likelihood = new
        best = np.copy()
    else:
        print('adding width did not help.')
    print('done add_width: node %d edge %d' % (len(h.nodes), len(h.edges)))
    return best, max_likelihood


def reduce_height(h, n):
    #print('before reduction: node %d edge %d' % (len(h.nodes), len(h.edges)))
    if n not in h.nodes:
        return h, -1
    g = h.copy()
    parents = list(g.predecessors(n))
    # choose the least reachable parent
    pfixes = what_to_fix(g, parents)
    pf = pfixes[0]
    print('fixing parent %d with %f' % (pf[0], pf[1]))
    grandparents = list(g.predecessors(pf[0]))
    if len(grandparents) == 0:
        print('got to the root')
        return g, -1
    # mix the siblings from the least reachable grand parent
    gpfixes = what_to_fix(g, grandparents)
    gpf = gpfixes[0]
    hp = merge_siblings_and_replace_parent(h, gpf[0])
    print('after reduction: node %d edge %d' % (len(hp.nodes), len(hp.edges)))
    return hp, gpf[0]


def merge_siblings_and_replace_parent(h, p):
    print('merging the children of %d' % p)
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


def update_likelihood_width(g, c, domains):
    likelihood = 0.0
    leaves = orgg.get_leaves(g)
    to_update = list(set(nx.descendants(g,c)).intersection(set(leaves)))
    for domain in domains:
        if domain in to_update:
            likelihood += orgh.local_log_likelihood(g, domain)
        else:
            for n in g.nodes:
                likelihood += math.log(g.node[n][domain['name']]['reach_prob_domain'])

    for p in g.nodes:
        g.node[p]['reach_prob'] = 0.0
        for domain in domains:
            g.node[p]['reach_prob'] += g.node[p][domain['name']]['reach_prob_domain']
    for p in to_update:
        g.node[p]['reach_prob'] = g.node[p]['reach_prob']/float(len(domains))

    return likelihood, g


def fix_node_agg_plus_plus(g, level, n, domains, likelihood):
    max_likelihood = likelihood
    gl = g.copy()
    print('starting with node %d and score %f' % (n, max_likelihood))
    best = gl.copy()
    # add an edge
    h, updated_gp = reduce_height(gl, n)
    if updated_gp == -1:
        return best, max_likelihood
    topo_order = list(nx.topological_sort(h))
    new, lp = update_likelihood_height(h, updated_gp, topo_order, domains)
    print('new score after the attempt to reduce height: %f' % new)
    if new > max_likelihood:
        print('improved by reducing height')
        max_likelihood = new
        best = lp.copy()
    else:
        print('reducing height did not help.')
    return best, max_likelihood


def find_another_parent(g, n, level):
    cands = dict()
    level_up = orgg.level_up(g, level)
    if len(level_up) == 0:
        return -1
    for c in level_up:
        cands[c] = orgh.get_transition_sim(g.node[n]['rep'], g.node[c]['rep'])
    scands = sorted(cands.items(), key=operator.itemgetter(1), reverse=True)
    cand = scands[0][0]
    if cand in list(g.predecessors(n)):
        print('picked one of the parents as a new one')
        return -1
    if cand in list(nx.descendants(g,n)):
        print('we do not want to add loop.')
        return -1
    return cand


def sort_nodes_sim(g, c, nodes):
    sims = dict()
    for n in nodes:
        sims[n] = orgh.get_transition_sim(g.node[n]['rep'], g.node[c]['rep'])
    ssims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True)
    return ssims










