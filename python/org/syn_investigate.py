import json
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


