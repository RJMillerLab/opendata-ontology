import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_train_probs():
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/train_dsps_multidim.json', 'r'))


    multidim_sps = list(multidim_tableprobs.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    print('number of tables: %d' % len(multidim_sps))


    xs = [i for i in range(len(multidim_sps))]
    plt.plot(xs, multidim_sps, color='r', label='multidim org (avg:'+'{:.3f}'.format(sum(multidim_sps)/len(multidim_sps))+')')
    #plt.plot(xs, flat, color='g', label='baseline (avg:'+'{:.3f}'.format(sum(flat)/len(flat))+')')

    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Open Data')
    plt.savefig('od_multidim_orgs_train.pdf')

    print('%f' % (sum(multidim_sps)/len(multidim_sps)))






plot_train_probs()

