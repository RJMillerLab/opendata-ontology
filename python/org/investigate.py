import json
import load as orgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot():
    before = []
    before100 = []
    after = []
    flat = []
    flat100 = []
    multidim = []
    for t, p in tableprobs_before.items():
        before.append(p)
        after.append(tableprobs_after[t])
        before100.append(tableprobs_before100[t])
        flat.append(tableprobs_flat[t])
        flat100.append(tableprobs_flat100[t])
        multidim.append(tableprobs_multidim[t])
    inx = np.argsort(np.array(before))
    after = list(np.array(after)[inx])
    before = list(np.array(before)[inx])
    before100 = list(np.array(before100)[inx])
    flat = list(np.array(flat)[inx])
    flat100 = list(np.array(flat100)[inx])
    multidim = list(np.array(multidim)[inx])

    print('%f -> %f -> %f -> %f -> %f -> %f' % (sum(flat)/len(flat), sum(before)/len(before), sum(after)/len(after), sum(multidim)/len(multidim), sum(flat100)/len(flat100), sum(before100)/len(before100)))
    xs = [i for i in range(len(before))]
    plt.plot(xs, before, color='r', label='clustering')
    #plt.plot(xs, before100, color='brown', label='clustering100')
    plt.plot(xs, after, color='b', label='fixed')
    plt.plot(xs, flat, color='g', label='baseline')
    plt.plot(xs, flat100, color='grey', label='baseline100')
    #plt.plot(xs, multidim, color='black', label='2-dimensional')
    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Benchmark')
    plt.savefig('org.pdf')



def test():
    print("Loading domains")
    adomains = list(orgl.add_ft_vectors(orgl.iter_domains()))
    print("Reduce tags")
    atags, atagdomains = orgl.reduce_tag_vectors(adomains)
    tagtables = dict()
    tabletags = dict()
    tagcounts = dict()
    for t, doms in atagdomains.items():
        for dom in doms:
            table = dom['name'][:dom['name'].rfind('_')]
            if table not in tabletags:
                tabletags[table] = []
            if t not in tabletags:
                tabletags[table].append(t)
            if t not in tagtables:
                tagtables[t] = []
                tagcounts[t] = 0
            if table not in tagtables:
                tagtables[t].append(table)
                tagcounts[t] += 1
    print('tableprobs: %d' % (len(tableprobs_before), len(tableprobs_after)))
    count = 0
    seen = []
    for t, p in tableprobs_before.items():
        if p < 0.05:
            count += 1
            print('%s - %f: %d' % (t, p, len(tabletags[t])))
            if t not in seen:
                seen.append(t)
                for g in tabletags[t]:
                    print('tag counts {}'.format([len(tagtables[g]) for g in tabletags[t]]))
                print('------------------')
    print('smalls: %d' % count)

    print('**********************')
    count = 0
    for t, p in tableprobs_before.items():
        if p > 0.5:
            count += 1
            print('%s - %f: %d' % (t, p, len(tabletags[t])))
            for g in tabletags[t]:
                print('tag counts {}'.format([len(tagtables[g]) for g in tabletags[t]]))
            print('------------------')
    print('bigs: %d' % count)



tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651flat_br.json', 'r'))
tableprobs_flat100 = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/flat_dists_2651_gamma100.json', 'r'))
#tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy.json', 'r'))
tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_f2op.json', 'r'))
tableprobs_before100 = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists2651fuzzy_gamma100.json', 'r'))
#tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_agg_singledim_2651_tag_dists.json', 'r'))
#tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single.json', 'r'))
tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single_f2op.json' , 'r'))
tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2_f2op.json', 'r'))
#tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_3.json', 'r'))
#tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2.json', 'r'))

plot()

