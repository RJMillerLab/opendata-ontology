import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    plt.xlabel('Semantic Group', fontsize=18)
    plt.xlim([0,len(ys)])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Measure', fontsize=18)
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
  #  plt.title('Classifiers on Data Lakes')
    plt.savefig('socrata_classifiers.pdf')
    plt.legend(loc='best', fancybox=True)
    plt.clf()


def plot_syn_classifiers_accuracy():
    results = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/model_results_365.json'))
    f1s, precs, recalls, accs = [], [], [], []
    for t, r in results.items():
        #if r['f1'] <= 0.45:
        #    continue
        f1s.append(r['f1'])
        precs.append(r['precision'])
        recalls.append(r['recall'])
        accs.append(r['accuracy'])
    print('classifiers: %d' % len(f1s))
    print(f1s)
    print(precs)
    print(recalls)
    inx = np.argsort(np.array(f1s))
    ys = np.array(f1s)[inx]
    plt.plot(ys, color='r', label='F1')
    ys = np.array(precs)[inx]
    plt.plot(ys, color='g', label='Precision')
    ys = np.array(recalls)[inx]
    plt.plot(ys, color='b', label='Recall')
    #plt.plot(accs, color='black')
    plt.xlabel('Semantic Group Classifier', fontsize=18)
    plt.xlim([0,len(ys)])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Measure', fontsize=18)
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
    #plt.title('Classifiers on Synthetic Benchmark')
    plt.savefig('synthetic_classifiers.pdf')
    plt.legend(loc='best', fancybox=True)
    plt.clf()


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
        ps.append(len(sls))
        bs.append(len(sbls))
        diffs.append(bs[-1]-ps[-1])


    inx = np.argsort(np.array(ps))
    ys = list(np.array(bs)[inx])
    plt.plot(ys, color='b', label='Boosted Grouping')
    ys = list(np.array(ps)[inx])
    plt.plot(ys, color='r', label='Available Grouping')
    plt.title('Boosted Semantic Grouping on Socrata Data Lakes')
    plt.xlabel('Table')
    plt.ylabel('Number of Semantic Groups')
    plt.xlim([0,len(ps)])
    plt.ylim([0, max(max(ps),max(bs))])
    plt.grid(linestyle="dotted")
    plt.legend(loc='best', fancybox=True)
    plt.savefig('table_label_boosted_socrata.pdf')
    plt.clf()


    print('ts: %d' % len(diffs))
    print('min: %d' % min(diffs))
    print('max: %d' % max(diffs))
    print(len([l for l in diffs if l >0]))



def plot_boosted_labels_other_tables():
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
        if len(sls) != 0:
            continue
        ps.append(len(sls))
        bs.append(len(sbls))
        diffs.append(bs[-1]-ps[-1])


    inx = np.argsort(np.array(bs))
    ys = list(np.array(bs)[inx])
    plt.plot(ys, color='b', label='Boosted Grouping')
    #ys = list(np.array(ps)[inx])
    #plt.plot(ys, color='r', label='Available Grouping')
    plt.title('Boosted Socrata Semantic Grouping on Non-Socrata Data Lakes')
    plt.xlabel('Table')
    plt.ylabel('Number of Semantic Groups')
    plt.xlim([0,len(ps)])
    plt.ylim([0,max(bs)+1])
    plt.grid(linestyle="dotted")
    #plt.legend(loc='best', fancybox=True)
    plt.savefig('table_label_boosted_non_socrata.pdf')
    plt.clf()


    print('ts: %d' % len(diffs))
    print('min: %d' % min(diffs))
    print('max: %d' % max(diffs))
    print(len([l for l in diffs if l >0]))


#plot_syn_classifiers_accuracy()
#plot_boosted_labels_socrata_tables()
#plot_boosted_labels_other_tables()
plot_classifiers_accuracy()
