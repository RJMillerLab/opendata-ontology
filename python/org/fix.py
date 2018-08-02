import org.hierarchy as orgh
import org.graph as orgg
import networkx as nx
import operator
import math
import numpy as np
import copy


sims = dict()

def fix_plus(g, domains, tagdomains):
    orgg.height(g)
    print('started fixing with %d domains.' % len(domains))
    iteration_success_probs = []
    init(g, domains)
    h, hf = g.copy(), g.copy()
    max_likelihood, gp, max_success_probs = orgh.get_success_prob(h, domains, tagdomains)
    initial_sp = max_likelihood
    initial_success_probs = copy.deepcopy(max_success_probs)
    best = gp.copy()
    level_n = list(orgg.get_leaves(gp))
    while len(level_n) > 1:
        print('len(level_n): %d nodes: %d edges: %d' % (len(level_n), len(gp.nodes), len(gp.edges)))
        hf, ll, sps, its = fix_level_plus(best.copy(), level_n, domains, max_likelihood, max_success_probs, tagdomains)
        iteration_success_probs.extend(list(its))
        print('after fix_level: node %d edge %d likelihood %f' % (len(hf.nodes), len(hf.edges), ll))
        imp = orgh.get_improvement(max_success_probs, sps)
        print('imp: %f' % imp)
        if imp > 0:
        #if ll > max_likelihood:
            print('improving after fixing level')
            max_likelihood = ll
            best = hf.copy()
            max_success_probs = copy.deepcopy(sps)
        level_n = orgg.level_up(gp, level_n)
    print('initial success prob: %f  and best success prob: %f' % (initial_sp, max_likelihood))
    print('improvement in success probs: %f' % orgh.get_improvement(initial_success_probs, max_success_probs))
    print('after fix_level: node %d edge %d' % (len(best.nodes), len(best.edges)))
    orgg.height(best)
    #ml, np, sps = orgh.get_success_prob(best, domains, tagdomains)
    #print('final success prob: %f' % ml)
    return best, iteration_success_probs


def fix_level_plus(g, level, domains, likelihood, success_probs, tagdomains):
    fixfunctions = [add_parent, change_parent]
    iteration_success_probs = []
    fixes = what_to_fix(g.copy(), level)
    max_likelihood = likelihood
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
            hp, newlikelihood, newsps, its = ffunc(best.copy(), level, f[0], domains, likelihood, max_success_probs, tagdomains)
            if newlikelihood < 0.0:
                continue
            #acceptance_ratio = min(1.0, newlikelihood/max_likelihood)
            #u = np.random.uniform(0.0, 1.0)
            #if u <= acceptance_ratio:
            #    if acceptance_ratio < 1.0:
            #        print('acceptance_ratio<1.0')
            #    print('accepted the operator with ratio %f. Old ll: %f New ll: %f' % (acceptance_ratio, likelihood, newlikelihood))
            #    best = hp.copy()
            #    max_likelihood = newlikelihood
            #    max_success_probs = copy.deepcopy(newsps)
            #    likelihood = newlikelihood
            #else:
            #    print('did not accept the operator: %f' % acceptance_ratio)
            iteration_success_probs.extend(list(its))
    return best, max_likelihood, max_success_probs, iteration_success_probs


def change_parent(g, level, n, domains, likelihood, success_probs, tagdomains):
    iteration_success_probs = []
    parents = list(g.predecessors(n))
    newparent = find_another_parent(g, n, level)
    if newparent == -1:
        return g, -1.0, [], []
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
    updates = list(ans)
    potentials = list(ans)
    h = update_graph_change_parent(g.copy(), oldparent, n, newparent)
    for a in ans:
        for d in list(g.predecessors(a)):
            potentials = list(set(potentials+list(nx.descendants(g, d))))

    new, np, sps = orgh.recompute_success_prob(h.copy(), domains, potentials, updates, tagdomains)
    iteration_success_probs.append(new)
    return np, new, sps, iteration_success_probs


def add_parent(g, level, n, domains, likelihood, success_probs, tagdomains):
    if n not in g.nodes:
        return g, -1.0, [], []
    iteration_success_probs = []
    gp = g.copy()
    max_likelihood = likelihood
    max_success_probs = copy.deepcopy(success_probs)
    best = gp.copy()
    choices = list((set(orgg.level_up(gp, level)).difference(set(nx.descendants(gp, n)))).difference(gp.predecessors(n)))
    schoices = sort_nodes_sim(g, n, choices)
    for sp in schoices:
        p = sp[0]
        h = gp.copy()
        ans1 = list(nx.ancestors(h, p))
        ans1.append(p)
        ans2 = list(nx.ancestors(h, n))
        ans2.append(n)
        ans1 = set(ans1)
        ans2 = set(ans2)
        updates = []
        if p in list(nx.ancestors(h, n)):
            ans = ans2.difference(ans1)
            ans = list(ans.union({n}))
            updates = [n]
        else:
            ans = ans1.difference(ans2)
            ans = list(ans.union({n, p}))
            updates = list(ans)
        potentials = list(ans)
        hap = update_graph_add_parent(h, p, n, tagdomains)
        for a in ans:
            for d in list(hap.predecessors(a)):
                t = set(potentials+list(nx.descendants(hap, d)))
                potentials = list(t)

        new, gl, sps = orgh.recompute_success_prob(hap.copy(), domains, potentials, updates, tagdomains)
        imp = orgh.get_improvement(max_success_probs, sps)
        print('imp: %f' % imp)
        acceptance_ratio = min(1.0, new/max_likelihood)
        u = np.random.uniform(0.0, 1.0)
        if u <= acceptance_ratio:
            if acceptance_ratio < 1.0:
                print('acceptance_ratio<1.0')
            print('accepted the operator with ratio %f. Old ll: %f New ll: %f' % (acceptance_ratio, likelihood, new))
            best = gl.copy()
            max_likelihood = new
            max_success_probs = copy.deepcopy(sps)
            likelihood = new
        else:
            print('did not accept the operator: %f' % acceptance_ratio)
        iteration_success_probs.append(max_likelihood)
        #if imp > 0:
        #if new > max_likelihood:
        #    print('connecting to %d improved' % p)
        #    max_likelihood = new
        #    max_success_probs = copy.deepcopy(sps)
        #    best = gl.copy()
        #    return best, max_likelihood, max_success_probs
    return best, max_likelihood, max_success_probs, iteration_success_probs


# incrementally updating log likelihood when adding an edge from p to c.
def update_likelihood_height(g, p, topo_order, domains):
    for domain in domains:
        tps = get_trans_prob(g, p, domain)
        for s in list(g.successors(p)):
            if domain['name'] not in g[p][s]:
                g[p][s][domain['name']] = dict()
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
        tsl.append(sims[s][domain['name']])
        ts[s] = sims[s][domain['name']]
    maxs = max(tsl)
    for s in sps:
        tps[s] = math.exp(ts[s]-maxs)
        d += tps[s]
    for s in sps:
        tps[s] = tps[s]/d
    return tps


def update_graph_add_parent(g, p, c, tagdomains):
    h = g.copy()
    leaves = orgg.get_leaves(h)
    to_update = []
    if p in list(nx.ancestors(g, c)):
        to_update = []
    else:
        to_update = list((set(nx.ancestors(h,p)).difference(nx.ancestors(h,c))).union({p}))
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


def reduce_height(h, n):
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
    #print('after reduction: node %d edge %d' % (len(hp.nodes), len(hp.edges)))
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










