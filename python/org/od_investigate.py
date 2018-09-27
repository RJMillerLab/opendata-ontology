import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot():
    before = []
    after = []
    #flat = []
    for t, p in tableprobs_before.items():
        before.append(p)
        after.append(tableprobs_after[t])
        #flat.append(tableprobs_flat[t])
    inx = np.argsort(np.array(after))
    after = list(np.array(after)[inx])
    before = list(np.array(before)[inx])
    #flat = list(np.array(flat)[inx])

    xs = [i for i in range(len(before))]
    plt.plot(xs, before, color='r', label='initial org (avg:'+'{:.3f}'.format(sum(before)/len(before))+')')
    plt.plot(xs, after, color='b', label='fixed org (avg:'+'{:.3f}'.format(sum(after)/len(after))+')')
    #plt.plot(xs, flat, color='g', label='baseline (avg:'+'{:.3f}'.format(sum(flat)/len(flat))+')')

    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Open Data')
    plt.savefig('od_multidim_orgs.pdf')

    #print('%f -> %f -> %f' % (sum(flat)/len(flat), sum(before)/len(before), sum(after)/len(after)))
    print('%f -> %f' % (sum(before)/len(before), sum(after)/len(after)))

tableprobs_flat = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/flat_dists_2651_g10.json', 'r'))
tableprobs_before = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/multidim_dists_before_1419_2.json', 'r'))
tableprobs_after = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/multidim_dists_1419_2_g10rhap.json', 'r'))

plot()

