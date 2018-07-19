import org.hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import math
import numpy as np


sims = dict()

def fix(g, domains):
    print('started fixing')
    gp = init(g, domains)
    level_n = orgg.get_leaves(gp)
    h = gp.copy()
    while len(level_n) > 1:
        print('len(level_n): %d edges: %d' % (len(level_n), len(h.edges)))
        for n in level_n:
            h = fix_level(h, level_n, domains)
            orgh.evaluate(h, domains)
        level_n = orgg.level_up(h, level_n)
    return h


def fix_level(g, level, domains):
    h = g.copy()
    fixes = what_to_fix(h, level)
    #for f in fixes:
    f = fixes[0]
    #print('fixing %d with %f' % (f[0], f[1]))
    h = fix_node(h, level, f[0], domains)
    return h


def fix_node(g, level, n, domains):
    max_score = orgh.log_likelihood(g, domains)
    #print('starting with score %f' % max_score)
    best = g
    choices = orgg.level_up(g, level)
    topo_order = list(nx.topological_sort(g))
    for p in choices:
        if p in g.predecessors(n):
            continue
        h = g.copy()
        h = update_graph(h, p, n)
        # pick a parent of n in this level to remove
        #to_remove = random.choice(list(set(g.predecessors(n)).intersection(set(level))))
        new = update_log_likelihood(h, n, p, topo_order, domains)
        print('new score: %f' % new)
        if new > max_score:
            print('improving')
            max_score = new
            best = h
            print('fix_node new edges: %d' % len(g.edges))
    return best


# incrementally updating log likelihood when adding an edge from p to c.
def update_log_likelihood(g, c, p, topo_order, domains):
    to_update = list(nx.descendants(g, p))

    for domain in domains:
        g[p][c][domain['name']] = dict()
        # trans prob of c and its siblings (of p) must be changed.
        for s in g.successors(p):
            g[p][s][domain['name']]['trans_prob_domain'] = get_trans_prob(g, p, s, domain)
            g.node[s][domain['name']]['reach_prob_domain'] = g.node[p][domain['name']]['reach_prob_domain'] * g[p][s][domain['name']]['trans_prob_domain']
        # initialize probs for a new domain
        for s in to_update:
            g.node[c][domain['name']]['reach_prob_domain'] = 0.0
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
            #print('reach prob of %s: %f' % (domain['name'], g.node[p][domain['name']]['reach_prob_domain']))
            g.node[p]['state_loglikelihood'] += math.log(g.node[p][domain['name']]['reach_prob_domain'], 10)
    for p in to_update:
        g.node[p]['reach_prob'] = g.node[p]['reach_prob']/float(len(domains))
    log_likelihood = 0.0
    for n in g.nodes:
        log_likelihood += g.node[p]['state_loglikelihood']
    return log_likelihood


def what_to_fix(g, nodes):
    m = {}
    for n in nodes:
        m[n] = g.node[n]['reach_prob']
    return sorted(m.items(), key=operator.itemgetter(1))


def init(g, domains):
    global sims
    h = calculate_states(g, domains)
    sims = get_trans_matrix(g, domains)
    return h


# calculate local log likelihood of the state and reachability probability given a domain
def calculate_states(g, domains):
    h = g.copy()
    for domain in domains:
        gp = orgh.get_domain_edge_probs(h, domain)
        gpp = orgh.get_domain_node_probs(gp)
        for n in gpp.nodes:
            if domain['name'] not in gpp.node[n]:
                gpp.node[n][domain['name']] = dict()
            if 'state_loglikelihood' not in gpp.node[n]:
                gpp.node[n][domain['name']]['reach_prob_domain'] = gpp.node[n]['reach_prob_domain']
                gpp.node[n]['state_loglikelihood'] = math.log(gpp.node[n]['reach_prob_domain'], 10)
                gpp.node[n]['reach_prob'] = gpp.node[n]['reach_prob_domain']
            else:
                gpp.node[n][domain['name']]['reach_prob_domain'] = gpp.node[n]['reach_prob_domain']
                gpp.node[n]['state_loglikelihood'] += math.log(gpp.node[n]['reach_prob_domain'], 10)
                gpp.node[n]['reach_prob'] += gpp.node[n]['reach_prob_domain']
        for e in gpp.edges:
            p, ch = e[0], e[1]
            if domain['name'] not in gpp[p][ch]:
                gpp[p][ch][domain['name']] = dict()
            gpp[p][ch][domain['name']]['trans_prob_domain'] = gpp[p][ch]['trans_prob_domain']
            gpp[p][ch][domain['name']]['trans_sim_domain'] = gpp[p][ch]['trans_sim_domain']
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


def get_trans_prob(g, p, c, domain):
    d = 0.0
    for s in g.successors(p):
        d += math.exp(sims[s][domain['name']])
    return math.exp(sims[c][domain['name']]) / d


def update_graph(g, p, c):
    h = g.copy()
    leaves = orgg.get_leaves(h)
    to_add = list(set(h.descendants(p)).intersection(set(leaves)).difference(set(h.descendants(c))))
    h.add_edge(p, c)
    # update population matrix
    for n in to_add:
        h.node[p]['population'].append(h.node[n]['rep'])
    # update rep vector
    h.node[p]['rep'] = list(np.mean(np.array(h.node[p]['population']), axis=0))
    return h









