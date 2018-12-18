import json
import math
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_boosted_labels_socrata_tables2():
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    boosted_table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/model_results.json'))
    # finding tables that are in our repo
    domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs', 'r'))
    tables = dict()
    for d in domains:
        t = d['name'][:d['name'].rfind('_')]
        if t not in tables:
            tables[t] = True

    # finding labels with high accuracy models
    goodlabels = []
    for t, r in results.items():
        if not label_names[str(t)].startswith('socrata'):
            continue
        if r['f1'] > 0.45:
            goodlabels.append(t)
    print(len(goodlabels))
    print(len(table_labels))
    print(len(boosted_table_labels))
    ps, bs, diffs = [], [], []
    for t, ls in table_labels.items():
        if t not in tables:
            continue
        sls = [l for l in ls if label_names[str(l)].startswith('socrata')]
        sbls = [l for l in boosted_table_labels[t] if label_names[str(l)].startswith('socrata')]# or str(l) in goodlabels]
        if len(sls) == 0 and len(sbls) == 0:
            continue
        ps.append(len(set(sls)))
        bs.append(len(set(sbls)))
        diffs.append(bs[-1]-ps[-1])

    table_num = len(ps)*1.0
    print('%d and %d' % (len(ps), len(bs)))
    hps = {u:(ps.count(u)/table_num) for u in set(ps)}
    hbs = {u:(bs.count(u)/table_num) for u in set(bs)}
    print(list(hps.keys()))
    print(list(hbs.keys()))
    ns = 5*np.array(list(hbs.keys()))
    width = 1.75
    plt.bar(ns+width, np.array(list(hbs.values())), width, label='boosted grouping', color='darkblue', hatch="//")
    ns = 5*np.array(list(hps.keys()))
    plt.bar(ns+2*width, np.array(list(hps.values())), width, label='original grouping', color='teal', hatch=".")
    plt.title('Semantic Grouping on Open Data Lake')
    plt.ylabel('Table Ratio')
    plt.xlabel('#Semantic Groups')
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


def get_booste_d_tables():
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    boosted_table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/model_results.json'))
    domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs', 'r'))
    tables = dict()
    for d in domains:
        t = d['name'][:d['name'].rfind('_')]
        if t not in tables:
            tables[t] = True


    # finding labels with high accuracy models
    goodlabels = []
    for t, r in results.items():
        if not label_names[str(t)].startswith('socrata'):
            continue
        if r['f1'] > 0.45:
            goodlabels.append(t)
    print(len(goodlabels))
    print(len(table_labels))
    print(len(boosted_table_labels))
    boosted = []
    diffs = []
    for t, ls in table_labels.items():
        sls = [l for l in ls if label_names[str(l)].startswith('socrata')]
        sbls = [l for l in boosted_table_labels[t] if label_names[str(l)].startswith('socrata')]# or str(l) in goodlabels]
        if len(sls)<len(sbls):
            boosted.append(t)
            diffs.append(len(sbls)-len(sls))
    print('boosted: %d' % len(set(boosted)))
    print('diffs: %f' % (sum(diffs)/float(len(diffs))))
    return list(set(boosted))


def get_nometa_tables():
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    boosted_table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/model_results.json'))
    domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs', 'r'))
    tables = dict()
    for d in domains:
        t = d['name'][:d['name'].rfind('_')]
        if t not in tables:
            tables[t] = True

    nometa_tables = dict()

    # finding labels with high accuracy models
    goodlabels = []
    for t, r in results.items():
        if not label_names[str(t)].startswith('socrata'):
            continue
        #if r['f1'] > 0.00001:
        goodlabels.append(label_names[t])
    print('goodlabels: %d' % len(goodlabels))
    print(len(table_labels))
    print(len(boosted_table_labels))
    for t, ls in table_labels.items():
        sls = [l for l in ls if label_names[str(l)].startswith('socrata')]
        sbls = [l for l in boosted_table_labels[t] if label_names[str(l)] in goodlabels]#.startswith('socrata')]
        if len(sls) == 0 and len(sbls)>0:
            nometa_tables[t] = True

    print('nometa: %d' % len(nometa_tables))
    good_label_tables = dict()
    good_tables = dict()
    # pick one table from each label
    for t, ls in boosted_table_labels.items():
        for l in ls:
            if label_names[l] in goodlabels and label_names[l] not in good_label_tables:
                good_label_tables[l] = t
                good_tables[t] = True
    for t, ls in boosted_table_labels.items():
        for l in ls:
            if label_names[l] in goodlabels:
                good_tables[t] = True
                if len(good_tables) == 1000:
                    break
            if len(good_tables) == 1000:
                break

    ls_nometa = dict()
    for t in good_tables.keys():
        for l in boosted_table_labels[t]:
            if label_names[l] in goodlabels:
                ls_nometa[l] = True
    print('ls_nometa: %d' % len(ls_nometa))


    json.dump(list(good_tables.keys()), open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/nometadata2.tables', 'w'))
    json.dump(goodlabels, open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/good_labels_names2.json', 'w'))



def plot_boosted_labels_socrata_tables():
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    boosted_table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/model_results.json'))
    domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs', 'r'))
    tables = dict()
    for d in domains:
        t = d['name'][:d['name'].rfind('_')]
        if t not in tables:
            tables[t] = True


    # finding labels with high accuracy models
    goodlabels = []
    for t, r in results.items():
        if not label_names[str(t)].startswith('socrata'):
            continue
        if r['f1'] > 0.45:
            goodlabels.append(t)
    print(len(goodlabels))
    print(len(table_labels))
    print(len(boosted_table_labels))
    ps, bs, diffs = [], [], []
    for t, ls in table_labels.items():
        sls = [l for l in ls if label_names[str(l)].startswith('socrata')]
        sbls = [l for l in boosted_table_labels[t] if label_names[str(l)].startswith('socrata')]# or str(l) in goodlabels]
        if len(sls) == 0 and len(sbls) == 0:
            continue
        ps.append(len(set(sls)))
        bs.append(len(set(sbls)))
        diffs.append(bs[-1]-ps[-1])


    table_num = len(ps)*1.0
    hhps = {u:(ps.count(u)/table_num) for u in set(ps) if u<=20}
    hhbs = {u:(bs.count(u)/table_num) for u in set(bs) if u<=20}
    print('min hps: %d' % min(hhps))
    print('min hbs: %d' % min(hhbs))
    print(list(hhps.keys()))
    print(list(hhbs.keys()))
    hps = copy.deepcopy(hhps)
    for k, v in hhbs.items():
        if k not in hps:
            hps[k] = 0
    hbs = copy.deepcopy(hhbs)
    for k, v in hhps.items():
        if k not in hbs:
            hbs[k] = 0
    print('%d and %d' % (len(hbs), len(hps)))
    print('min %d %d' % (min(list(hhbs.keys())), min(list(hhps.keys()))))
    print('0: %d b %d' % (hps[1], hbs[1]))

    ks = list(hbs.keys())
    ks.sort()
    ns = np.array([5*k for k in ks])
    width = 1.5

    ys = [hps[k] for k in ks]
    plt.bar(ns+width, ys, width, label='original grouping', color='teal')
    ys = [hbs[k] for k in ks]
    plt.bar(ns, ys, width, label='boosted grouping', color='darkblue')

    plt.title('Semantic Grouping on Open Data Lake')
    plt.ylabel('Table Ratio')
    plt.xticks([ns[i] for i in range(0, len(ns),4)], [ks[i] for i in range(0,len(ks),4)], ha='center')
    plt.xlabel('#Semantic Groups Per Table')
    plt.ylim([0.0,0.3])
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
    plt.savefig('table_label_boosted_socrata_top.pdf')
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
    plt.xlabel('Semantic Group Classifier')
    plt.xlim([0,len(ys)])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Measure')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
    plt.title('Enhancing Semantic Groupings to Open Data Lake')
    plt.savefig('socrata_classifiers.pdf')
    plt.legend(loc='best', fancybox=True)
    plt.clf()

def plot_test_probs_boosted():
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/test_dsps_multidim.json', 'r'))
    boost_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_boosted_output/test_dsps_multidim.json',    'r'))

    boosted_tables= get_booste_d_tables()

    multidim_sps = []
    boost_sps = []
    opt_sps = []
    for t, p in multidim_tableprobs.items():
        if t in boosted_tables:
            multidim_sps.append(p)
            boost_sps.append(boost_tableprobs[t])
    list(multidim_tableprobs.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    inx = np.argsort(np.array(boost_sps))
    boost_sps = list(np.array(boost_sps)[inx])
    inx = np.argsort(np.array(opt_sps))
    opt_sps = list(np.array(opt_sps)[inx])
    print('number of tables: %d' % len(multidim_sps))
    print('number of tables: %d' % len(boost_sps))
    print('number of tables: %d' % len(opt_sps))

    print('zeros in multidim: %d' % multidim_sps.count(0.0))
    print('zeros in boost: %d' % boost_sps.count(0.0))


    xs = [i for i in range(len(multidim_sps))]
    #plt.plot(xs, multidim_sps, color='darkblue', label='multidim org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(multidim_sps)/len(multidim_sps))+')', marker='x', markevery=300, markersize=4)
    plt.plot(xs, multidim_sps, color='darkblue', label='multidim org', marker='x', markevery=300, markersize=4)
    #plt.plot(xs, boost_sps, color='crimson', label='boosted org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(boost_sps)/len(boost_sps))+')', marker='+', markevery=300, markersize=4)
    plt.plot(xs, boost_sps, color='crimson', label='boosted org', marker='+', markevery=300, markersize=4)
    plt.plot(xs, opt_sps, color='crimson', label='boosted org', marker='+', markevery=300, markersize=4)


    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Table')
    plt.ylabel('Discovery Probability')
    #plt.title('Table Discovery in Open Data Lake')
    plt.savefig('boost_sps.pdf')
    plt.clf()

    print('multidim %f' % (sum(multidim_sps)/len(multidim_sps)))
    print('boost %f' % (sum(boost_sps)/len(boost_sps)))



def plot_test_probs():
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/test_dsps_multidim.json', 'r'))
    flat_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/results/all_flat_0_table_sps.json', 'r'))
    boost_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_boosted_output/test_dsps_multidim.json',    'r'))
    opt_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_opt/test_dsps_opt.json', 'r'))

    multidim_sps = list(multidim_tableprobs.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    flat_sps = list(flat_tableprobs.values())
    inx = np.argsort(np.array(flat_sps))
    flat_sps = list(np.array(flat_sps)[inx])
    boost_sps = list(boost_tableprobs.values())
    inx = np.argsort(np.array(boost_sps))
    boost_sps = list(np.array(boost_sps)[inx])
    opt_sps = list(opt_tableprobs.values())
    inx = np.argsort(np.array(opt_sps))
    opt_sps = list(np.array(opt_sps)[inx])
    print('number of tables: %d' % len(multidim_sps))
    print('number of tables: %d' % len(flat_sps))
    print('number of tables: %d' % len(boost_sps))
    print('number of tables: %d' % len(opt_sps))

    print('zeros in multidim: %d' % multidim_sps.count(0.0))
    print('zeros in boost: %d' % boost_sps.count(0.0))
    print('zeros in flat %d' % flat_sps.count(0.0))

    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    #table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    ys = []

    lts = dict()
    for t, ls in table_labels.items():
        for l in ls:
            if not label_names[l].startswith('socrata'):
                continue
            if l not in lts:
                lts[l] = []
            if t not in lts[l]:
                lts[l].append(t)
    brs = [len(lts[l]) for l, ts in lts.items()]
    print(len(brs))
    print('brs: %d' % sum(r>5 for r in brs))


    for t, p in multidim_tableprobs.items():
        if p > 0.001:
            continue
        tls = [l for l in table_labels[t] if label_names[str(l)].startswith('socrata')]
        ys.append(len(tls))
    ys.sort()
    print('no tag: %d' % sum(i < 2 for i in ys))
    print(len(ys))
    plt.plot([i for i  in range(len(ys))], ys)
    plt.savefig('lhs.pdf')
    plt.clf()

    bs1 = [t for t, p  in multidim_tableprobs.items() if p < 0.001]
    fs = [t for t, p  in flat_tableprobs.items() if p<0.001]
    bs2 = [t for t, p  in boost_tableprobs.items() if p<0.001]
    print('bs1: %d' % len(bs1))
    print('fs: %d' % len(fs))
    print('bs2: %d' % len(bs2))
    print('bs: %d' % len(set(fs).intersection(set(bs1).intersection(set(bs2)))))

    xs = [i for i in range(len(multidim_sps))]
    #plt.plot(xs, multidim_sps, color='darkblue', label='multidim org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(multidim_sps)/len(multidim_sps))+')', marker='x', markevery=300, markersize=4)
    plt.plot(xs, multidim_sps, color='darkblue', label='multidim org', marker='x', markevery=300, markersize=4)
    #plt.plot(xs, flat_sps, color='teal', label='flat org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(flat_sps)/len(flat_sps))+')', marker='+', markevery=300, markersize=4)
    plt.plot(xs, flat_sps, color='teal', label='flat org', marker='+', markevery=300, markersize=4)
    #plt.plot(xs, boost_sps, color='crimson', label='boosted org ($P(\mathcal{T}|\mathcal{O})=$'+'{:.3f}'.format(sum(boost_sps)/len(boost_sps))+')', marker='+', markevery=300, markersize=4)
    plt.plot(xs, opt_sps, color='crimson', label='opt org', marker='+', markevery=300, markersize=4)




    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Table', fontsize=16)
    plt.ylabel('Discovery Probability', fontsize=16)
    #plt.title('Table Discovery in Open Data Lake')
    plt.savefig('od_multidim_flat_org_all.pdf')
    plt.clf()

    print('multidim %f' % (sum(multidim_sps)/len(multidim_sps)))
    print('flat %f' % (sum(flat_sps)/len(flat_sps)))
    print('opt %f' % (sum(opt_sps)/len(opt_sps)))




def plot_train_probs():
    multidim_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/train_dsps_multidim.json', 'r'))
    multidim_update_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/test_dsps_multidim.json', 'r'))
    flat_tableprobs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/flat_0_table_sps.json', 'r'))

    multidim_update_sps = [p for t, p in multidim_update_tableprobs.items() if t in multidim_tableprobs]
    inx = np.argsort(np.array(multidim_update_sps))
    multidim_update_sps = list(np.array(multidim_update_sps)[inx])
    multidim_sps = list(multidim_tableprobs.values())
    inx = np.argsort(np.array(multidim_sps))
    multidim_sps = list(np.array(multidim_sps)[inx])
    flat_sps = [p for t, p in flat_tableprobs.items() if t in multidim_tableprobs]
    #flat_sps = list(flat_tableprobs.values())
    inx = np.argsort(np.array(flat_sps))
    flat_sps = list(np.array(flat_sps)[inx])
    print('number of tables: %d' % len(multidim_sps))
    print('number of all tables: %d' % len(multidim_update_tableprobs))


    xs = [i for i in range(len(multidim_sps))]
    plt.plot(xs, multidim_sps, color='darkblue', label='multidim ($P(\mathcal{T}|\mathcal{O})=$'+'{:.4f}'.format(sum(multidim_sps)/len(multidim_sps))+')', marker='x', markevery=300, markersize=4)

    #plt.plot(xs, multidim_update_sps, color='crimson', label='multidim after repo update ($P(\mathcal{T}|\mathcal{O})=$'+'{:.4f}'.format(sum(multidim_update_sps)/len(multidim_update_sps))+')', linestyle='--')

    plt.plot(xs, flat_sps, color='teal', label='baseline ($P(\mathcal{T}|\mathcal{O})=$'+'{:.4f}'.format(sum(flat_sps)/len(flat_sps))+')', marker='+', markevery=300, markersize=4)

    plt.legend(loc='best', fancybox=True)
    plt.grid(linestyle='dotted')
    plt.xlabel('Table')
    plt.ylabel('Discovery Probability')
    #plt.title('Table Discovery after Updating Open Data Lake')
    plt.savefig('od_multidim_flat_orgs_train.pdf')
    plt.clf()

    print('multidim %f' % (sum(multidim_sps)/len(multidim_sps)))
    print('test multidim %f' % (sum(multidim_update_sps)/len(multidim_update_sps)))
    print('flat %f' % (sum(flat_sps)/len(flat_sps)))



def plot_table_dists():
    domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs', 'r'))
    #domains = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/domain_embs'))
    tables = dict()
    labeltables = dict()
    tablelabels = dict()
    labeldoms =  dict()
    for d in domains:
        t = d['name'][:d['name'].rfind('_')]
        if t not in tables:
            tables[t] = []
        if d['name'] not in tables[t]:
            tables[t].append(d['name'])
        if not d['tag'].startswith('socrata_'):
            continue
        l = d['tag']
        if l not in labeldoms:
            labeldoms[l] = []
        if d['name'] not in labeldoms[l]:
            labeldoms[l].append(d['name'])
        if t not in tablelabels:
            tablelabels[t] = []
        if l not in tablelabels[t]:
            tablelabels[t].append(l)
        if l not in labeltables:
            labeltables[l] = []
        if t not in labeltables[l]:
            labeltables[l].append(t)
    print('tables: %d' % len(tables))
    print('labeltables: %d' % len(labeltables))
    print('tablelabels: %d' % len(tablelabels))

    ys = [math.log(len(ds)) for t, ds in tables.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    #plt.plot(xs, ys, color='darkblue')
    plt.scatter(xs, ys, color='crimson')
    #plt.title('Attributes in Tables', fontsize=14)
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Table', fontsize=18)
    plt.ylabel('#Attributes (Log)', fontsize=18)
    plt.grid(linestyle="dotted")
    plt.savefig('tableatts_dist_log.pdf')
    plt.clf()

    ys = [math.log(len(ds)) for t, ds in labeldoms.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    #plt.plot(xs, ys, color='darkblue')
    plt.scatter(xs, ys, color='darkgreen')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Semantic Group', fontsize=18)
    plt.ylabel('#Positive Samples (Log)', fontsize=18)
    plt.grid(linestyle="dotted")
    plt.savefig('labelatts_dist_log.pdf')
    plt.clf()



    ys = [math.log(len(ds)) for t, ds in tablelabels.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    #plt.plot(xs, ys, color='darkblue')
    plt.scatter(xs, ys, color='darkblue')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Table', fontsize=18)
    plt.ylabel('#Semantic Groups (Log)', fontsize=18)
    plt.grid(linestyle="dotted")
    plt.savefig('tablelabels_dist_log.pdf')
    plt.clf()


    ys = [math.log(len(ds)) for t, ds in labeltables.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    #plt.plot(xs, ys, color='darkblue')
    plt.scatter(xs, ys, color='darkblue')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Semantic Group', fontsize=14)
    plt.ylabel('#Tables (Log)', fontsize=14)
    plt.grid(linestyle="dotted")
    plt.savefig('labeltables_dist_log.pdf')
    plt.clf()


def plot_nometa_probs():
    nometa_probs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/nometa_output/fix_train_probs_0.json', 'r'))
    ys = list(nometa_probs.values())
    ys.sort()
    plt.plot([i for i in range(len(ys))], ys, marker='x', markevery=300, markersize=4, color='darkblue')
    plt.grid(linestyle="dotted")
    plt.ylabel('Discovery Probability', fontsize=18)
    plt.xlabel('Table', fontsize=18)
    #plt.title('Table Discovery in CKAN Data Lake by Boosting Metadata', fontsize=14)
    plt.savefig('ckan_nometa_sps.pdf')
    plt.clf()


def plot_boosted_label_distribution_ckan():
    nometa_tables = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/nometadata2.tables','r'))
    table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'))
    boosted_table_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/tables.boosted_labels'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'))
    good_labels  = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/socrata/good_labels_names2.json', 'r'))

    bs = []
    lts = dict()
    for t, ls in table_labels.items():
        if t not in nometa_tables:
            continue
        sbls = [l for l in boosted_table_labels[t] if label_names[str(l)] in good_labels]
        for l in ls:
            if label_names[str(l)] in good_labels:
                if l not in lts:
                    lts[l] = []
                if t not in lts[l]:
                    lts[l].append(t)

        bs.append(len(set(sbls)))
    ml = ''
    mc = 0
    print(len(lts))
    for l, ts in lts.items():
        if len(ts)>mc:
            mc=len(ts)
            ml = label_names[str(l)]
    print(ml)
    table_num = len(bs)*1.0
    print('table num %d %d' % (len(nometa_tables), len(bs)))
    hbs = {u:(bs.count(u)/table_num) for u in set(bs)}
    print(list(hbs.keys()))
    print(hbs[1])
    labels = [x for x in list(hbs.keys())]
    labels.sort()
    labels = labels[:20]
    ys = [hbs[x] for x in labels]
    xs = [i+1 for i in range(len(ys))]
    print(max(ys))
    plt.bar(xs, ys, color='teal')
    plt.xticks(labels, ha='center')
    #plt.title('Semantic Grouping Added to CKAN Data Lake', fontsize=14)
    plt.ylabel('Table Ratio', fontsize=18)
    plt.xlabel('#Semantic Groups Per Table', fontsize=18)
    plt.xlim([0, max(labels)+1])
    plt.ylim([0.0,0.4])
    plt.grid(linestyle="dotted")
    plt.savefig('table_label_boosted_ckan.pdf')
    plt.clf()




#plot_boosted_label_distribution_ckan()

#plot_table_dists()
#plot_train_probs()
#plot_boosted_labels_socrata_tables()
plot_classifiers_accuracy()
#plot_test_probs()
#get_nometa_tables()
#plot_nometa_probs()
#plot_test_probs_boosted()
#get_booste_d_tables()
