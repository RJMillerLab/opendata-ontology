import json
import operator
import numpy as np
import networkx as nx

#TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_embs'
#tag_embs = json.load(open(TAG_EMB_FILE, 'r'))
tag_embs = dict()


def init_fromfile(tagemb_filename):
    global tag_embs
    tag_embs = json.load(open(tagemb_filename, 'r'))

def init_fromarrays(keys, vecs):
    global tag_embs
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
            parentsems.extend(statesems[p])
        parentsems = list(set(parentsems))
        cent = get_state_rep(tags, parentsems)
        statesems[state] = cent

    print('ss %d' % len(statesems))

    edge_num = int(lines[state_num+1])
    for i in range(state_num+2, state_num+edge_num):
        ps = lines[i].split(':')
        sfile.write(statesems[ps[0]] + '->' + statesems[ps[1]] + '\n')

    hfile.close()
    sfile.close()
    print('sematic of org in %s.' % sem_filename)



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

