import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import json

def discoverable_nodes(n):
    return list(nx.descendants(g, n))

def evaluate_query(root, s):
    paths = nx.all_simple_paths(g, root, s, cutoff)
    prob = 0.0
    for p in paths:
        path_prob = 1.0
        for i in range(len(p)-1):
            path_prob *= edges[p[i]][p[i+1]]
        prob += path_prob
    print('paths: %d  %f' % (len(list(paths)), prob))
    return prob


def evaluate_root_att(root):
    probs = dict()
    qs = discoverable_nodes(root)
    #print('discoverable_nodes: %d' % len(qs))
    for q in qs:
        #if q in round_seen_atts:
            #print('node already visited')
        #    continue
        round_seen_atts[q] = True
        if root in seen_paths:
            if q in seen_paths[root]:
                probs[q] = seen_paths[root][q]
                #print('using cached paths.')
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
    round_probs_intersect = dict()
    count = 0
    for t, cs in tcs.items():
        if t not in round_probs:
            round_probs[t] = 0.0
            round_probs_intersect[t] = 1.0
        if t not in table_rounds:
            continue
        print('start: %s' % t)
        count += 1
        if count%20==0:
            print('round %d' % count)
        print('columns of root: %d' % len(cs))
        for c in cs:
            ps = evaluate_root_att(c)
            for q, p in ps.items():
                tname = get_table_name(q)
                if tname not in round_probs:
                    round_probs[tname] = 0.0
                    round_probs_intersect[tname] = 1.0
                round_probs[tname] += p
                round_probs_intersect[tname] *= p
                #if round_probs[tname] > 1.0:
                #    print('> 1.0')
    for t, p in round_probs.items():
        round_probs[t] = (p-round_probs_intersect[t])/float(len(table_rounds))
    print(round_probs)
    print('tables: %d' % len(tcs))
    print('evaluated tables: %d' % len(round_probs))
    print('round_seen_atts %d' % len(round_seen_atts))
    print('g.nodes: %d' % len(g.nodes()))
    json.dump(round_probs, open(NAVIGATION_PROBS, 'w'))
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


def plot(ps):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ps.sort()
    ax.set_ylim([0, 1.0])
    ax.plot([i for i in range(len(ps))], ps)
    plt.tight_layout()
    plt.savefig("ps.pdf")
    plt.close()



TABLE_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_atts'
JACC_EDGE_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/jacc_edges'
NAVIGATION_PROBS = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/navigation_probs'
cutoff = 10
g = nx.DiGraph()
edges = json.load(open(JACC_EDGE_FILE, 'r'))

seen_paths = dict()
round_seen_atts = dict()

init()
print('edges: %d' % len(g.edges()))
ts = json.load(open(TABLE_ATTS, 'r'))
SIMULATION_NUM = int(len(ts)/10)
print('tables: %d' % len(ts))
print('SIMULATION_NUM: %d' % SIMULATION_NUM)
ps = evaluate()
plot(list(ps.values()))

print('done')
