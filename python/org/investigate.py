import json
import load as orgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot():
    before = []
    after = []
    flat = []
    #flat100 = []
    multidim = []
    for t, p in tableprobs_before.items():
        before.append(p)
        after.append(tableprobs_after[t])
        flat.append(tableprobs_flat[t])
        #flat100.append(tableprobs_flat100[t])
        multidim.append(tableprobs_multidim[t])
    inx = np.argsort(np.array(multidim))
    after = list(np.array(after)[inx])
    before = list(np.array(before)[inx])
    flat = list(np.array(flat)[inx])
    #flat100 = list(np.array(flat100)[inx])
    multidim = list(np.array(multidim)[inx])

    xs = [i for i in range(len(before))]
    plt.plot(xs, before, color='r', label='initial org (avg:'+'{:.3f}'.format(sum(before)/len(before))+')')
    plt.plot(xs, after, color='b', label='fixed org (avg:'+'{:.3f}'.format(sum(after)/len(after))+')')
    plt.plot(xs, flat, color='g', label='baseline (avg:'+'{:.3f}'.format(sum(flat)/len(flat))+')')
    #plt.plot(xs, flat100, color='grey', label='baseline100')
    plt.plot(xs, multidim, color='black', label='2-dimensional (avg:'+'{:.3f}'.format(sum(multidim)/len(multidim))+')')
    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Benchmark')
    plt.savefig('bigbench_multidim_orgs.pdf')

    print('%f -> %f -> %f -> %f' % (sum(flat)/len(flat), sum(before)/len(before), sum(after)/len(after), sum(multidim)/len(multidim)))


# looking into tables in the LHS of plots.
def look_lhs():
    rhs = []
    for t, p in tableprobs_multidim.items():
        if tableprobs_before[t] <= tableprobs_multidim[t]:
            rhs.append(t)
    print('rhs multidim: %d' % len(rhs))

    rhs = []
    for t, p in tableprobs_after.items():
        if tableprobs_before[t] <= tableprobs_after[t]:
            rhs.append(t)
    print('rhs singledim: %d' % len(rhs))

    lhs = []
    #for t, p in tableprobs_before.items():
    #    if tableprobs_before[t] > tableprobs_after[t]:
    for t, p in tableprobs_multidim.items():
        if tableprobs_after[t] > tableprobs_multidim[t] and tableprobs_before[t] > tableprobs_multidim[t]:
            lhs.append(t)
    print('lhs: %d' % len(lhs))
    print("Loading domains")
    adomains = list(orgl.add_ft_vectors(orgl.iter_domains()))
    print("Reduce tags")
    atags, atagdomains = orgl.reduce_tag_vectors(adomains)
    tagtables = dict()
    tabletags = dict()
    tagcounts = dict()
    tabledoms = dict()
    for t, doms in atagdomains.items():
        for dom in doms:
            colid = int(dom['name'][dom['name'].rfind('_')+1:])
            table = dom['name'][:dom['name'].rfind('_')]+'_'+str(colid%2)
            if table not in tabledoms:
                tabledoms[table] = 0
            tabledoms[table] += 1
            if table not in tabletags:
                tabletags[table] = []
            if t not in tabletags[table]:
                tabletags[table].append(t)
            if t not in tagtables:
                tagtables[t] = []
                tagcounts[t] = 0
            if table not in tagtables[t]:
                tagtables[t].append(table)
                tagcounts[t] += 1
    sings = 0
    stags = dict()
    for t in lhs:
        if tabledoms[t] == 1:
            sings += 1
        for g in tabletags[t]:
            stags[g] = len(tagtables[g])
        print('%s: ds: %d ts: %d' % (t, tabledoms[t], len(tabletags[t])))
    print('lhs sings: %d' % sings)
    sings = 0
    for t, d in tabledoms.items():
        if d == 1:
            sings += 1
    print('all sings: %d' % sings)
    print(stags)


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



#tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651flat_br.json', 'r'))
tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651_g10.json', 'r'))
tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/flat_dists_2651_notor.json', 'r'))
tableprobs_flat100 = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/flat_dists_2651_gamma100.json', 'r'))
#tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy.json', 'r'))
#tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_f2op.json', 'r'))
#tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists2651fuzzy_f1opap.json', 'r'))
tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_g10t75frhap.json', 'r'))
tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists2651fuzzy_g10t752opint.json'))
#tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_agg_singledim_2651_tag_dists.json', 'r'))
#tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single.json', 'r'))
#tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single_f2op.json' , 'r'))
#tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists_2651_single_f1opap.json' , 'r'))
tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_agg_singledim_g10t75frhap_2651_tag_dists.json', 'r'))
tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists_2651_single_g10t752opint.json', 'r'))
tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2_f2op.json', 'r'))
tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_2651_2_g10rhap.json', 'r'))
#tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2_g10rhap.json', 'r'))
#tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_3.json', 'r'))
#tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2.json', 'r'))

#plot()

look_lhs()
