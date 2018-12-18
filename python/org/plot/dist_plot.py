import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(jsonfile, pdffile, title):
    results = json.load(open(jsonfile, 'r'))
    ps = []
    for t, p in results.items():
        ps.append(p)
    print('max: %d' % max(ps))
    print('min: %d' % min(ps))
    weights = np.ones_like(ps)/float(len(ps))
    plt.hist(ps, bins=list(np.linspace(0.0,1.0,500)), weights=weights)
    plt.xlabel('Table Reachability Probability')
    plt.title(title)
    #plt.xlim([0.0,max(ps)])
    plt.xlim([-0.1,1.1])
    plt.ylim([plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])#[0.0,0.2])
    plt.savefig(pdffile)
    plt.clf()

def plot_probs(jsonfile, pdffile, title):
    results = json.load(open(jsonfile, 'r'))
    probs = list(results.values())
    probs.sort()
    plt.plot(probs)
    plt.xlabel('Table')
    plt.ylabel('Reachability Prob')
    plt.title(title)
    plt.savefig(pdffile)
    plt.clf()



def plot_fix(jsonfile, pdffile, label, title):
    print(pdffile)
    results = json.load(open(jsonfile, 'r'))
    plt.plot(results)
    plt.xlabel('Iterations')
    plt.ylabel(label)
    plt.title(title)
    plt.savefig(pdffile)
    plt.clf()


def double_plot(jsonfile1, pdffile1, title1, jsonfile2, pdffile2, title2):
    results1 = json.load(open(jsonfile1, 'r'))
    ps1 = [p for t, p in results1.items()]
    print('max: %f min: %f' % (max(ps1), min(ps1)))

    results2 = json.load(open(jsonfile2, 'r'))
    ps2 = [p for t, p in results2.items()]
    print('max: %f min: %f' % (max(ps2), min(ps2)))

    xlim = max(max(ps1), max(ps2))

    weights1 = np.ones_like(ps1)/float(len(ps1))
    plt.hist(ps1, bins=list(np.linspace(0.0,xlim,1000)), weights=weights1)
    plt.xlabel('Domain Reachability Probability')
    plt.title(title1)
    plt.xlim([0.0,xlim])
    plt.ylim([0.0, plt.gca().get_ylim()[1]])
    plt.savefig(pdffile1)
    plt.clf()

    weights2 = np.ones_like(ps2)/float(len(ps2))
    plt.hist(ps2, bins=list(np.linspace(0.0,xlim,1000)), weights=weights2)
    plt.xlabel('Domain Reachability Probability')
    plt.title(title2)
    plt.xlim([0.0,xlim])
    plt.ylim([0.0, plt.gca().get_ylim()[1]])
    plt.savefig(pdffile2)
    plt.clf()


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

    ys = [len(ds) for t, ds in tables.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='darkblue')
    #plt.title('Attributes in Tables', fontsize=14)
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xlabel('Table', fontsize=14)
    plt.ylabel('Number of Attributes', fontsize=14)
    plt.grid(linestyle="dotted")
    plt.savefig('tableatts_dist.pdf')
    plt.clf()

    ys = [len(ds) for t, ds in labeldoms.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='darkblue')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xlabel('Semantic Group', fontsize=14)
    plt.ylabel('Number of Attributes/Positive Samples', fontsize=14)
    plt.grid(linestyle="dotted")
    plt.savefig('labelatts_dist.pdf')
    plt.clf()



    ys = [len(ds) for t, ds in tablelabels.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='darkblue')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xlabel('Table', fontsize=14)
    plt.ylabel('Number of Semantic Groups', fontsize=14)
    plt.grid(linestyle="dotted")
    plt.savefig('tablelabels_dist.pdf')
    plt.clf()


    ys = [len(ds) for t, ds in labeltables.items()]
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    plt.plot(xs, ys, color='darkblue')
    plt.xlim([0,len(xs)])
    plt.ylim([0,max(ys)])
    plt.xlabel('Semantic Group', fontsize=14)
    plt.ylabel('Number of Tables', fontsize=14)
    plt.grid(linestyle="dotted")
    plt.savefig('labeltables_dist.pdf')
    plt.clf()




def plot_label_table_dist():
    LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'
    TABLE_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'
    label_names = json.load(open(LABELS_FILE, 'r'))
    table_labels = json.load(open(TABLE_LABELS_FILE, 'r'))
    labeltables = dict()
    for t, ls in table_labels.items():
        for l in ls:
            if not label_names[str(l)].startswith('socrata_'):
                continue
            if l not in labeltables:
                labeltables[l] = []
            if t not in labeltables[l]:
                labeltables[l].append(t)
    xs, ys = [], []
    print('labels: %d' % len(labeltables))
    for l, ts in labeltables.items():
        ys.append(len(ts))
    ys.sort(reverse=True)
    xs = [i for i in range(len(ys))]
    plt.plot(xs[50:], ys[50:], color='darkblue')
    plt.title('Table Associations to Semantic Groups', fontsize=14)
    plt.xlim([0.0,len(xs[50:])])
    plt.ylim([0.0,max(ys[50:])])
    plt.xlabel('Table ID', fontsize=14)
    plt.ylabel('Number of Semantic Groups', fontsize=14)
    plt.grid(linestyle="dotted")
    plt.savefig('tablegroups_dist.pdf')
    plt.clf()




plot_table_dists()
#plot_label_table_dist()
