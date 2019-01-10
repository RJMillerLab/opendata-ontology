import networkx as nx
import random
import json

def discoverable_nodes(n):
    return list(nx.descendants(g, n))

def evaluate_query(root, s):
    paths = nx.all_simple_paths(g, root, s, cutoff)
    print('paths: %d' % len(list(paths)))
    prob = 0.0
    for p in paths:
        path_prob = 1.0
        for i in range(len(p)-1):
            path_prob *= edges[p[i]][p[i+1]]
        prob += path_prob
    return prob


def evaluate_root_att(root):
    probs = dict()
    qs = discoverable_nodes(root)
    print('discoverable_nodes: %d' % len(qs))
    for q in qs:
        if q in round_seen_atts:
            continue
        round_seen_atts[q] = True
        if root in seen_paths:
            if q in seen_paths[root]:
                probs[q] = seen_paths[root][q]
                continue
        else:
            seen_paths[root] = dict()
        prob = evaluate_query(root, q)
        probs[q] = prob
        if q not in seen_paths:
            seen_paths[q] = dict()
        seen_paths[q][root] = prob
        seen_paths[root][q] = prob
    return probs

def evaluate():
    tcs = json.load(open(TABLE_ATTS, 'r'))
    ts = list(tcs.keys())
    rand_starts = random.sample(range(0,len(tcs)), SIMULATION_NUM)
    table_rounds = [ts[i] for i in rand_starts]
    round_probs = dict()
    count = 0
    for t, cs in tcs.items():
        if t not in table_rounds:
            continue
        count += 1
        if count%20==0:
            print('round %d' % count)
        print('cs: %d' % len(cs))
        for c in cs:
            ps = evaluate_root_att(c)
            for q, p in ps.items():
                tname = get_table_name(q)
                if tname not in round_probs:
                    round_probs[tname] = 0.0
                round_probs[tname] += p
                if round_probs[tname] > 1.0:
                    print('> 1.0')
    #for t, p in round_probs.items():
    #    round_probs[t] = p/float(len(table_rounds))
    print(round_probs)
    json.dump(round_probs, NAVIGATION_PROBS)
    return round_probs


def get_table_name(col_name):
    return col_name[:col_name.rfind('_')]

def init():
    sts = []
    for s, ts in edges.items():
        for t, p in ts.items():
            sts.append((s,t))

    g.add_edges_from(sts)

    # add all isolated nodes
    nodes = list(g.nodes())
    tcs = json.load(open(TABLE_ATTS, 'r'))
    for t, cs in tcs.items():
        for c in cs:
            if c not in nodes:
                g.add_node(c)



TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts'
JACC_EDGE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/jacc_edges'
NAVIGATION_PROBS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/navigation_probs'
SIMULATION_NUM = 10
cutoff = 10
g = nx.DiGraph()
edges = json.load(open(JACC_EDGE_FILE, 'r'))

seen_paths = dict()
round_seen_atts = dict()

init()
print('edges: %d' % len(g.edges()))
evaluate()

print('done')
