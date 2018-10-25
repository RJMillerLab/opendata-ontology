import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_boosted_labels_socrata_tables():
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    boosted_table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/model_results.json'))
    # finding labels with high accuracy models
    goodlabels = []
    for t, r in results.items():
        if r['f1'] > 0.45:
            goodlabels.append(t)
    print(len(goodlabels))
    print(len(table_labels))
    print(len(boosted_table_labels))
    ps, bs, diffs = [], [], []
    for t, ls in table_labels.items():
        sls = [l for l in ls if label_names[str(l)].startswith('socrata')]
        sbls = [l for l in boosted_table_labels[t] if label_names[str(l)].startswith('socrata') or str(l) in goodlabels]
        if len(sls) == 0 and len(sbls) == 0:
            continue
        ps.append(len(set(sls)))
        bs.append(len(set(sbls)))
        diffs.append(bs[-1]-ps[-1])

    table_num = len(ps)*1.0
    print('%d and %d' % (len(ps), len(bs)))
    hps = {u:(ps.count(u)/table_num) for u in set(ps)}
    hbs = {u:(bs.count(u)/table_num) for u in set(bs)}
    ns = 5*np.array(list(hbs.keys()))
    width = 1.75
    plt.bar(ns+width, np.array(list(hbs.values())), width, label='boosted grouping', color='darkblue', hatch="//")
    ns = 5*np.array(list(hps.keys()))
    plt.bar(ns+2*width, np.array(list(hps.values())), width, label='original grouping', color='teal', hatch=".")
    plt.title('Semantic Grouping on Open Data Lakes')
    plt.xlabel('Table Ratio in the Repository')
    plt.ylabel('Number of Semantic Groups')
    plt.xlim([min(min(list(hbs.keys())),min(list(hps.keys()))),max(max(list(hbs.keys())),max(list(hps.keys())))])
    plt.ylim([0.0,0.3])
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
    plt.savefig('table_label_boosted_socrata.pdf')
    plt.clf()



    print('ts: %d' % len(diffs))
    print('min: %d' % min(diffs))
    print('max: %d' % max(diffs))
    print(len([l for l in diffs if l >0]))
    print(sum(diffs))



def plot_classifiers_accuracy():
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/model_results.json'))
    f1s, precs, recalls, accs = [], [], [], []
    for t, r in results.items():
        if r['f1'] <= 0.45:
            continue
        f1s.append(r['f1'])
        precs.append(r['precision'])
        recalls.append(r['recall'])
        accs.append(r['accuracy'])
    print('classifiers: %d' % len(f1s))
    inx = np.argsort(np.array(f1s))
    ys = np.array(f1s)[inx]
    plt.plot(ys, color='r', label='F1')
    ys = np.array(precs)[inx]
    plt.plot(ys, color='g', label='Precision')
    ys = np.array(recalls)[inx]
    plt.plot(ys, color='b', label='Recall')
    #plt.plot(accs, color='black')
    plt.xlabel('Semantic Group')
    plt.xlim([0,len(ys)])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Measure')
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
    plt.title('Classifiers on Data Lakes')
    plt.savefig('socrata_classifiers.pdf')
    plt.legend(loc='best', fancybox=True)
    plt.clf()

def plot_test_probs():
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/test_dsps_multidim.json', 'r'))

    multidim_sps = list(multidim_tableprobs.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    print('number of tables: %d' % len(multidim_sps))


    xs = [i for i in range(len(multidim_sps))]
    plt.plot(xs, multidim_sps, color='darkblue', label='multidim org (avg:'+'{:.3f}'.format(sum(multidim_sps)/len(multidim_sps))+')', marker='x', markevery=300, markersize=4)

    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Open Data')
    plt.savefig('od_multidim_org_test.pdf')

    print('multidim %f' % (sum(multidim_sps)/len(multidim_sps)))




def plot_train_probs():
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/train_dsps_multidim.json', 'r'))
    flat_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/flat_0_table_sps.json', 'r'))

    multidim_sps = list(multidim_tableprobs.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    flat_sps = list(flat_tableprobs.values())
    inx = np.argsort(np.array(flat_sps))
    flat_sps = list(np.array(flat_sps)[inx])
    print('number of tables: %d' % len(multidim_sps))


    xs = [i for i in range(len(multidim_sps))]
    plt.plot(xs, multidim_sps, color='darkblue', label='multidim org (avg:'+'{:.3f}'.format(sum(multidim_sps)/len(multidim_sps))+')', marker='x', markevery=300, markersize=4)
    #xs = [i for i in range(len(flat_sps))]
    #plt.plot(xs, flat_sps, color='g', label='baseline (avg:'+'{:.3f}'.format(sum(flat_sps)/len(flat_sps))+')')

    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Tables')
    plt.ylabel('Discovery Probability')
    plt.title('Table Discovery in Open Data')
    plt.savefig('od_multidim_flat_orgs_train.pdf')

    print('multidim %f' % (sum(multidim_sps)/len(multidim_sps)))
    print('flat %f' % (sum(flat_sps)/len(flat_sps)))






#plot_train_probs()
plot_boosted_labels_socrata_tables()
#plot_test_probs()
