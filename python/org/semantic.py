import json
import operator
import numpy as np
import networkx as nx

tag_embs, tag_tables = dict(), dict()


def init_fromfile(tagemb_filename):
    global tag_embs, tag_tables
    tag_embs = json.load(open(tagemb_filename, 'r'))

def init_fromarrays(keys, vecs, i_tagtables):
    global tag_embs, tagtables
    tagtables = i_tagtables
    for i in range(len(keys)):
        tag_embs[keys[i]] = vecs[i]

def get_states_semantic(hierarchy_filename):
    hfile = open(hierarchy_filename, 'r')
    lines = hfile.read().splitlines()
    state_num = int(lines[0])
    for i in range(1, state_num+1):
        line = lines[i]
        print('processing state: %s' % line.split(':')[0])
        tags = line.split(':')[1].split('|')
        print('tags: ')
        print(tags)
        cent = get_state_rep(tags)
        print(cent)
        print('---------------')


def get_org_semantic(org_filename, sem_filename):
    print('tag_embs: %d' % len(tag_embs))
    sfile = open(sem_filename,'w')
    hfile = open(org_filename, 'r')
    lines = hfile.read().splitlines()
    state_num = int(lines[0])

    edges = []
    edge_num = int(lines[state_num+1])
    for i in range(state_num+2, state_num+edge_num):
        ps = lines[i].split(':')
        edges.append((ps[0], ps[1]))

    g = nx.DiGraph()
    g.add_edges_from(edges)
    top = list(nx.topological_sort(g))

    print('states in org: %d aned edges: %d' % (len(g.nodes), len(g.edges)))
    statestags = dict()

    for i in range(1, state_num+1):
        line = lines[i]
        state = line.split(':')[0]
        tags = line.split(':')[1].split('|')
        statestags[state] = tags

    statesems = dict()
    for s in top:
        tags = statestags[s]
        parentsems = []
        for p in g.predecessors(s):
            parentsems.extend(statesems[str(p)])
        parentsems = list(set(parentsems))
        cent = get_state_rep(tags, parentsems)
        statesems[str(s)] = cent

    print('ss %d' % len(statesems))

    edge_num = int(lines[state_num+1])
    for i in range(state_num+2, state_num+edge_num):
        ps = lines[i].split(':')
        sfile.write(statesems[ps[0]] + '->' + statesems[ps[1]] + '\n')

    hfile.close()
    sfile.close()
    print('sematic of org in %s.' % sem_filename)

def get_state_rep_freq(state_tags):
    if len(state_tags) == 1:
        return state_tags[0]
    tag_weights = {t:len(tag_tables[t]) for t in state_tags}
    sorted_tag_weights = sorted(tag_weights.items(), key=operator.itemgetter(1))
    return sorted_tag_weights[-1][0]


def get_state_rep_btree(state_tags):
    state_tags = [t for t in state_tags]
    if len(state_tags) == 1:
        return state_tags[0]
    centroid = get_centroid(state_tags)
    sims = {}
    for t in state_tags:
        sims[t] = get_similarity(centroid, tag_embs[t])
    ssims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True)
    return ssims[0][0]



def get_state_rep(state_tags, exclude_sems):
    if len(state_tags) == 1:
        return state_tags[0]
    centroid = get_centroid(state_tags)
    sims = {}
    for t in state_tags:
        sims[t] = get_similarity(centroid, tag_embs[t])
    ssims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True)
    print('the rep of state is %s with score %f' % (ssims[0][0], ssims[0][1]))
    for i in range(len(sims)):
        if ssims[i][0] not in exclude_sems:
            return ssims[i][0]
    return ssims[0][0]


def get_similarity(vec1, vec2):
    dp = np.dot(vec1,vec2)
    s = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return max(0.000001, s)


def get_centroid(state_tags):
    vecs = np.array([tag_embs[t] for t in state_tags])
    return np.mean(vecs, axis=0)

def org_with_semantic(org_filename, sem_filename):
    print('tag_embs: %d' % len(tag_embs))
    with open(org_filename, 'r') as hfile:
        lines = hfile.read().splitlines()
    state_num = int(lines[0])
    print('state_num: %d' % state_num)
    edges = []
    edge_num = int(lines[state_num+1])
    for i in range(state_num+2, state_num+edge_num):
        ps = lines[i].split(':')
        edges.append((ps[0], ps[1]))

    # building and propagating the semantics of states
    g = nx.DiGraph()
    g.add_edges_from(edges)
    print('cycle exists?')
    print(list(nx.simple_cycles(g)))
    top = list(nx.topological_sort(g))
    top.reverse()

    print('states in org: %d aned edges: %d' % (len(g.nodes), len(g.edges)))
    statestags = dict()

    for i in range(1, state_num+1):
        line = lines[i]
        state = line.split(':')[0]
        tags = line.split(':')[1].split('|')
        statestags[state] = tags

    leaves = list(set([x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)>0]))
    statesems = dict()
    statecents = dict()
    print('top: %d' % len(top))
    for s in top:
        statesems[s] = []
        if s in leaves:
            statesems[s].append(statestags[s][0])
            statecents[s] = statestags[s][0]
            continue
        for u in g.successors(s):
            cent = get_state_rep_freq(statesems[u], tag_tables)
            #cent = get_state_rep_btree(statesems[u])
            statecents[u] = cent
            statesems[s].append(cent)

    for n in g.nodes:
        g.node[n]['sem'] = '|'.join(list(set(statesems[n])))
    return g


def get_org_semantic_btree(org_filename, sem_filename):
    print('tag_embs: %d' % len(tag_embs))
    sfile = open(sem_filename,'w')
    with open(org_filename, 'r') as hfile:
        lines = hfile.read().splitlines()
    print('lines: %d' % len(lines))
    state_num = int(lines[0])
    print('state_num: %d' % state_num)
    edges = []
    edge_num = int(lines[state_num+1])
    for i in range(state_num+2, state_num+edge_num):
        ps = lines[i].split(':')
        edges.append((ps[0], ps[1]))

    # building and propagating the semantics of states
    g = nx.DiGraph()
    g.add_edges_from(edges)
    print(list(nx.simple_cycles(g)))
    top = list(nx.topological_sort(g))
    top.reverse()

    print('states in org: %d aned edges: %d' % (len(g.nodes), len(g.edges)))
    statestags = dict()

    for i in range(1, state_num+1):
        line = lines[i]
        state = line.split(':')[0]
        tags = line.split(':')[1].split('|')
        statestags[state] = tags

    leaves = list(set([x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)>0]))
    statesems = dict()
    statecents = dict()
    print('top: %d' % len(top))
    for s in top:
        statesems[s] = []
        if s in leaves:
            statesems[s].append(str(s) + ':' + statestags[s][0])
            statecents[s] = str(s) + ':' + statestags[s][0]
            continue
        for u in g.successors(s):
            cent = get_state_rep_btree(statesems[u])
            statecents[u] = str(u) + ':' + cent
            statesems[s].append(str(u) + ':' + cent)

    print('ss %d' % len(statesems))
    print('sc %d' % len(statecents))
    print('nodes %d' % len(g.nodes()))

    top = list(nx.topological_sort(g))
    for n in top:
        if n not in statecents:
            # root
            sfile.write('root->' + '|'.join(list(set(statesems[n])))+'\n')
            continue
        if n in leaves:
            sfile.write(statecents[n] + '->leaf\n')
            continue
        sfile.write(statecents[n] + '->' + '|'.join(list(set(statesems[n])))+'\n')

    #edge_num = int(lines[state_num+1])
    #for i in range(state_num+2, state_num+edge_num):
    #    ps = lines[i].split(':')
    #    sfile.write('|'.join(statesems[ps[0]]) + '->' + '|'.join(statesems[ps[1]]) + '\n')

    sfile.close()
    print('sematic of org in %s.' % sem_filename)


def organization_semantics(org_filename, sem_filename):
    print('tag_embs: %d' % len(tag_embs))
    hfile = open(org_filename, 'r')
    lines = hfile.read().splitlines()
    state_num = int(lines[0])

    edges = []
    edge_num = int(lines[state_num+1])
    for i in range(state_num+2, state_num+edge_num):
        ps = lines[i].split(':')
        edges.append((ps[0], ps[1]))

    g = nx.DiGraph()
    g.add_edges_from(edges)
    top = list(nx.topological_sort(g))

    print('states in org: %d aned edges: %d' % (len(g.nodes), len(g.edges)))
    statestags = dict()

    for i in range(1, state_num+1):
        line = lines[i]
        state = line.split(':')[0]
        tags = line.split(':')[1].split('|')
        statestags[state] = tags

    statesems = dict()
    for s in top:
        tags = statestags[s]
        parentsems = []
        for p in g.predecessors(s):
            parentsems.extend(statesems[str(p)])
        parentsems = list(set(parentsems))
        cent = get_state_rep(tags, parentsems)
        statesems[str(s)] = cent

    print("statesems")
    print(statesems)
    return statesems

