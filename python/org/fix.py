import org.hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import math
import numpy as np


sims = dict()

def fix(g, domains):
    print('started fixing')
    init(g, domains)
    level_n = list(orgg.get_leaves(g))
    h = g.copy()
    hf = g.copy()
    max_score, gp = orgh.log_likelihood(h, domains)
    best = gp
    curr_ll = max_score
    while len(level_n) > 1:
        print('len(level_n): %d nodes: %d edges: %d' % (len(level_n), len(h.nodes), len(h.edges)))
        hf, curr_ll = fix_level(gp, level_n, domains, curr_ll)
        print('after fix_level: node %d edge %d likelihood %f' % (len(hf.nodes), len(hf.edges), curr_ll))
        if curr_ll > max_score:
            print('improving after fixing level')
            max_score = curr_ll
            best = hf
            orgh.evaluate(hf, domains)
        level_n = orgg.level_up(hf, level_n)
    print('best likelihood: %f' % max_score)
    return best


def fix_level(g, level, domains, likelihood):
    h = g.copy()
    fixes = what_to_fix(h, level)
    #for f in fixes:
    f = fixes[0]
    print('fixing %d with %f' % (f[0], f[1]))
    #h = fix_node(h, level, f[0], domains)
    hp, likelihood = fix_node_agg(h, level, f[0], domains, likelihood)
    return hp.copy(), likelihood


def fix_node(g, level, n, domains):
    gp = g.copy()
    h, bp = add_width(gp, level, n, domains)
    return h


def add_width(g, level, n, domains):
    print('before add_width: node %d edge %d' % (len(g.nodes), len(g.edges)))
    if n not in g.nodes:
        print('%d not in the graph - width')
        return g
    gp = g.copy()
    max_score, gp = orgh.log_likelihood(gp, domains)
    print('starting with score %f' % max_score)
    best = gp
    choices = orgg.level_up(gp, level)
    #topo_order = list(nx.topological_sort(gp))
    bestparent = -1
    for p in choices:
        if p in gp.predecessors(n):
            print('we do not want loop in the dag.')
            continue
        h = gp.copy()
        print('update graph')
        h = update_graph(h, p, n)
        print('ll computation')
        new, gl = update_likelihood_width(h, p, domains)
        if new > max_score:
            print('improving')
            max_score = new
            bestparent = p
            best = gl
            print('fix_node new edges: %d' % len(h.edges))
            gp = h
    print('after add_width: node %d edge %d' % (len(best.nodes), len(best.edges)))
    return best, bestparent


# incrementally updating log likelihood when adding an edge from p to c.
def update_likelihood_height(g, p, topo_order, domains):
    print('printing update_log_likelihood')
    for domain in domains:
        tps = get_trans_prob(g, p, domain)
        for s in g.successors(p):
            if domain['name'] not in g[p][s]:
                g[p][s][domain['name']] = dict()
            g[p][s][domain['name']]['trans_prob_domain'] = tps[s]

    to_update = list(nx.descendants(g, p))

    for domain in domains:
        for s in g.successors(p):
            g.node[s][domain['name']]['reach_prob_domain'] = g.node[p][domain['name']]['reach_prob_domain'] * g[p][s][domain['name']]['trans_prob_domain']
        for s in to_update:
            g.node[s][domain['name']]['reach_prob_domain'] = 0.0
        for u in topo_order:
            for ch in g.successors(u):
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
    for s in g.successors(p):
        tps[s] = math.exp(sims[s][domain['name']])
        d += tps[s]
    for s, v in tps.items():
        tps[s] = v/d
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


def update_graph(g, p, c):
    h = g.copy()
    h.add_edge(p, c)
    leaves = orgg.get_leaves(h)
    to_add = list((set(nx.descendants(h,c)).intersection(set(leaves))).            difference(set(nx.descendants(h,p)).intersection(set(leaves))))
    to_update = list(nx.ancestors(h,p))
    to_update.append(p)
    for n in to_update:
        # update population matrix
        for a in to_add:
            h.node[n]['population'].append(h.node[a]['rep'])
        # update rep vector
        h.node[n]['rep'] = list(np.mean(np.array(h.node[n]['population']), axis=0))
    return h


def fix_node_agg(g, level, n, domains, likelihood):
    #gp = g.copy()
    #max_score, gl = orgh.log_likelihood(gp, domains)
    max_score = likelihood
    gl = g.copy()
    print('starting with node %d and score %f' % (n, max_score))
    best = gl.copy()
    h, updated_gp = reduce_height(gl, n)
    topo_order = list(nx.topological_sort(h))
    new, lp = update_likelihood_height(h, updated_gp, topo_order, domains)
    #new = orgh.log_likelihood(h, domains)
    print('new score after the attempt to reduce height: %f' % new)
    if new > max_score:
        print('improved by reducing height')
        max_score = new
        best = lp.copy()
    else:
        print('reducing height did not help.')
    #print('reduction done: node %d edge %d' % (len(h.nodes), len(h.edges)))
    hp = best.copy()
    h, bestparent = add_width(hp, level, n, domains)
    if bestparent == -1:
        return best, max_score
    new, np = update_likelihood_width(h, bestparent, domains)
    print('new score after the attempt to add width: %f' % new)
    if new > max_score:
        print('improved by add width')
        max_score = new
        best = np.copy()
    else:
        print('adding width did not help.')
    print('done add_width: node %d edge %d' % (len(h.nodes), len(h.edges)))
    return best, max_score


def reduce_height(h, n):
    #print('before reduction: node %d edge %d' % (len(h.nodes), len(h.edges)))
    if n not in h.nodes:
        print('%d not in the graph - height')
        return h
    g = h.copy()
    parents = list(g.predecessors(n))
    # choose the least reachable parent
    pfixes = what_to_fix(g, parents)
    pf = pfixes[0]
    print('fixing parent %d with %f' % (pf[0], pf[1]))
    grandparents = list(g.predecessors(pf[0]))
    if len(grandparents) == 0:
        print('got to the root')
        return g
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
        for n in h.successors(s):
            h.add_edge(p, n)
        h.remove_edge(p, s)
        if len(list(h.predecessors(s))) == 0:
            print('removing %d' % s)
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












