import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_test_probs():
    boost_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_boosted_output/test_dsps_multidim.json', 'r'))
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/test_dsps_multidim.json', 'r'))

    mtps = dict()
    for t, p in boost_tableprobs.items():
        if t not in multidim_tableprobs:
            print('table not found')
        else:
            mtps[t] = multidim_tableprobs[t]
    print('number of tables: %d' % len(multidim_tableprobs))
    multidim_sps = list(mtps.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    boost_sps = list(boost_tableprobs.values())
    inx = np.argsort(np.array(boost_sps))
    boost_sps = list(np.array(boost_sps)[inx])
    print('number of tables: %d' % len(boost_sps))


    xs = [i for i in range(len(multidim_sps))]
    plt.plot(xs, multidim_sps, color='darkblue', label='multidim org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(multidim_sps)/len(multidim_sps))+')', marker='x', markevery=300, markersize=4)
    plt.plot(xs, boost_sps, color='teal', label='boost org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(boost_sps)/len(boost_sps))+')', marker='+', markevery=300, markersize=4)



    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Table')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Open Data Lake')
    plt.savefig('od_multidim_boost_org_all.pdf')
    plt.clf()

    print('multidim %f' % (sum(multidim_sps)/len(multidim_sps)))
    print('boosted %f' % (sum(boost_sps)/len(boost_sps)))






#plot_train_probs()
plot_test_probs()
