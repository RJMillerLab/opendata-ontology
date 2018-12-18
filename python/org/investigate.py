import json
import load as orgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot():
    # loading files
    tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651_notor.json', 'r'))
    tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_g10t752opint.json'))
    tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single_g10t752opint.json', 'r'))
    tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_prune_rep_sim_threshold062_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_boost = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_boosted_2651_2_g10rhap.json', 'r'))


    before = []
    after = []
    flat = []
    #flat100 = []
    multidim = []
    multidim_prune = []
    multidim_boost = []
    for t, p in tableprobs_before.items():
        before.append(p)
        after.append(tableprobs_after[t])
        flat.append(tableprobs_flat[t])
        #flat100.append(tableprobs_flat100[t])
        multidim.append(tableprobs_multidim[t])
        multidim_prune.append(tableprobs_multidim_prune[t])
        multidim_boost.append(tableprobs_multidim_boost[t])
    print('loaded')
    inx = np.argsort(np.array(after))
    after = list(np.array(after)[inx])
    print('s1')
    inx = np.argsort(np.array(before))
    before = list(np.array(before)[inx])
    print('s2')
    inx = np.argsort(np.array(flat))
    flat = list(np.array(flat)[inx])
    print('s3')
    inx = np.argsort(np.array(multidim))
    multidim = list(np.array(multidim)[inx])
    print('s4')
    inx = np.argsort(np.array(multidim_prune))
    multidim_prune = list(np.array(multidim_prune)[inx])
    print('s5')
    inx = np.argsort(np.array(multidim_boost))
    multidim_boost = list(np.array(multidim_boost)[inx])


    xs = [i for i in range(len(before))]
    plt.plot(xs, before, color='r', label='initial org (avg:'+'{:.3f}'.format(sum(before)/len(before))+')')
    plt.plot(xs, after, color='b', label='fixed org (avg:'+'{:.3f}'.format(sum(after)/len(after))+')')
    plt.plot(xs, flat, color='g', label='baseline (avg:'+'{:.3f}'.format(sum(flat)/len(flat))+')')
    #plt.plot(xs, flat100, color='grey', label='baseline100')
    plt.plot(xs, multidim, color='black', label='2-dimensional (avg:'+'{:.3f}'.format(sum(multidim)/len(multidim))+')')
    plt.plot(xs, multidim_prune, color='orange', label='2-dimensional prune (avg:'+'{:.3f}'.format(sum(multidim_prune)/len(multidim_prune))+')')
    plt.plot(xs, multidim_boost, color='purple', label='2-dimensional prune (avg:'+'{:.3f}'.format(sum(multidim_boost)/len(multidim_boost))+')')
    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Benchmark')
    plt.savefig('bigbench_multidim_boosted_orgs.pdf')

    print('%f -> %f -> %f -> %f -> %f -> %f' % (sum(flat)/len(flat), sum(before)/len(before), sum(after)/len(after), sum(multidim)/len(multidim), sum(multidim_prune)/len(multidim_prune), sum(multidim_boost)/len(multidim_boost)))


# looking into tables in the LHS of plots.
def look_lhs():
    rhs = []
    for t, p in tableprobs_multidim.items():
        if p == 0.0:
            print('0.0 prob')
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



def plot_stats():
    stats = syn_multidim_stats
    domprune = []
    repprune = []
    for st in stats:
        domprune.append(st['active_domains'])
        repprune.append(st['active_reps'])
    xs = [i for i in range(len(domprune))]
    domprune.sort()
    repprune.sort()
    num_domains = max(domprune)
    noprune = [num_domains for i in xs]
    plt.plot(xs, domprune, color='blue', label='Prunning Domains')
    plt.plot(xs, repprune, color='green', label='Prunning by Reps')
    plt.plot(xs, noprune, color='red', linestyle='--', label='No Prunning')
    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Exploration Iteration')
    plt.ylabel('Number of Considered Data Points')
    plt.title('Prunning Domains and States in Synthetic Benchmark')
    plt.savefig('synthetic_prunning_dom_rep.pdf')

def plot_synthetic_label_dom_sims():
    tagdomsims = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/rep_domain_sims.json', 'r'))
    ys = []
    for t, dsims in tagdomsims.items():
        for d, s in dsims.items():
            ys.append(s)
    ys.sort()
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='blue')
    plt.xlabel('label-domain pairs')
    plt.ylabel('similarity')
    plt.savefig('syntehtic_labeldomain_sims.pdf')
    plt.clf()




def plot_od_label_dom_sims():
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json', 'r'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json', 'r'))
    table_label_names = dict()
    for t, ls in table_labels.items():
        table_label_names[t] = []
        for l in ls:
            table_label_names[t].append(label_names[str(l)])
    ys = []
    doms = dict()
    for label, domsims in label_dom_sims.items():
        for dom, sim in domsims.items():
            if dom not in doms:
                doms[dom] = 0
            doms[dom] += 1
            table = dom[:dom.rfind('_')]
            if label in table_label_names[table]:
                ys.append(sim)
    print('doms: %d  labels: %d  pairs: %d' % (len(doms), len(label_dom_sims), sum(list(doms.values()))))
    xs = [i for i in range(len(ys))]
    ys.sort()
    plt.plot(xs, ys, color='blue')
    plt.xlabel('label-domain pairs')
    plt.ylabel('similarity')
    plt.savefig('od_labeldomain_sims.pdf')
    plt.clf()


def plot_od_dom_dom_sims():
    dom_sims = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json', 'r'))
    ys = []
    for dom, dsims in dom_sims.items():
        ys.extend(list(dsims.values()))
    ys.sort()
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='blue')
    plt.xlabel('domain-domain pairs')
    plt.ylabel('similarity')
    plt.savefig('od_domaindomain_sims.pdf')


def init():
    label_dom_sims = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_domain_sims.json', 'r'))
    syn_multidim_stats = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_1773_prunning_stats.json', 'r'))


    #tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651flat_br.json', 'r'))
    tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651_g10.json', 'r'))
    tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651_notor.json', 'r'))
    tableprobs_flat100 = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/flat_dists_2651_gamma100.json', 'r'))
    #tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy.json', 'r'))
    #tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_f2op.json', 'r'))
    #tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists2651fuzzy_f1opap.json', 'r'))
    tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_g10t75frhap.json', 'r'))
    tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists2651fuzzy_g10t752opint.json'))
    #tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_agg_singledim_2651_tag_dists.json', 'r'))
    #tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single.json', 'r'))
    #tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single_f2op.json' , 'r'))
    #tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists_2651_single_f1opap.json' , 'r'))
    tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/fix_agg_singledim_g10t75frhap_2651_tag_dists.json', 'r'))
    tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/agg_dists_2651_single_g10t752opint.json', 'r'))
    tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2_f2op.json', 'r'))
    tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_2651_2_g10rhap.json', 'r'))
    #tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2_g10rhap.json', 'r'))
    #tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_3.json', 'r'))
    #tableprobs_multidim = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_2651_2.json', 'r'))
    tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_partial_sim_threshold_2651_2_g10rhap.json', 'r'))
    #tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_partial_sim_threshold06_2651_2_g10rhap.json', 'r'))
    #tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_partial_sim_threshold06_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_partial_sim_threshold062_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_prune = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/multidim_dists_prune_rep_sim_threshold062_2651_2_g10rhap.json', 'r'))
    tableprobs_multidim_boost = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_boosted_2651_2_g10rhap.json', 'r'))

plot()
#plot_stats()
#look_lhs()
#plot_od_label_dom_sims()
#plot_od_dom_dom_sims()
#plot_od_label_dom_sims()
#plot_synthetic_label_dom_sims()
