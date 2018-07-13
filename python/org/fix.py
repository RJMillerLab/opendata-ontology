import org.hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import math


def fix(g, domains):
    print('started fixing')
    gp = init(g, domains)
    level_n = orgg.get_leaves(gp)
    h = gp.copy()
    while len(level_n) > 0:
        print('len(level_n): %d' % len(level_n))
        for n in level_n:
            h = fix_level(h, level_n, domains)
        level_n = orgg.level_up(h, level_n)
    return h


def fix_level(g, level, domains):
    h = g.copy()
    fixes = what_to_fix(h, level)
    #for f in fixes:
    f = fixes[0]
    print('fixing %d with %f' % (f[0], f[1]))
    h = fix_node(h, level, f[0], domains)
    return h


def fix_node(g, level, n, domains):
    max_score = orgh.log_likelihood(g, domains)
    print('starting with score %f' % max_score)
    best = g
    choices = orgg.level_up(g, level)
    topo_order = list(nx.topological_sort(g))
    for p in choices:
        if p not in g.predecessors(n):
            continue
        h = g.copy()
        h.add_edge(p, n)
        # pick a parent of n in this level to remove
        #to_remove = random.choice(list(set(g.predecessors(n)).intersection(set(level))))
        new = update_log_likelihood(h, n, p, topo_order, domains)
        print('new score: %f' % new)
        if new > max_score:
            print('improving')
            max_score = new
            best = h
    return best


# incrementally updating log likelihood when adding an edge from p to c.
def update_log_likelihood(g, c, p, topo_order, domains):
    descs = list(nx.descendants(g, c))
    descs.append(c)
    to_update = descs

    for domain in domains:
        g.node[c][domain['name']]['reach_prob_domain'] += g.               node[p][domain['name']]['reach_prob_domain']*                                       g[p][c][domain['name']]['trans_sim_domain']
        for u in topo_order:
            for ch in g.successors(u):
                if ch not in to_update:
                    continue
                g.node[ch][domain['name']]['reach_prob_domain'] += g.node[u][domain['name']]['reach_prob_domain']*g[u][ch][domain['name']]['trans_sim_domain']

    for p in to_update:
        g.node[p]['state_loglikelihood'] = 0.0
        for domain in domains:
            g.node[p]['state_loglikelihood'] += math.log(g.node[p][domain['name']]['reach_prob_domain'], 10)
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
    h = calculate_states(g, domains)
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



