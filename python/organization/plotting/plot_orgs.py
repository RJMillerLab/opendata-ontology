import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("acm-2col.mplstyle")

ORG_FILE = "/home/fnargesian/FINDOPENDATA_DATASETS/10k/orgs.results"
FIG_FILE = "orgs.pdf"
orgs = []
intradimoverlap = []
entropy = []
nonuniformity = []
density = []

def read_organizations(orgFilename):
    # orgs consists of two lists, each corresponds a dimension
    # each entry in a dimension list a tuple where the first element is label_id and
    # the second element is label_name
    of = open(orgFilename, 'r')
    lines = of.read().splitlines()
    for i in range(int(len(lines)/3)):
        line1 = lines[3*i]
        line2 = lines[3*i+1]
        line3 = lines[3*i+2]
        parts = line1.split("|")
        os = []
        os.append([(int(parts[2*j]),parts[2*j+1]) for j in range(int(len(parts)/2))])
        parts = line2.split("|")
        os.append([(int(parts[2*j]),parts[2*j+1]) for j in range(int(len(parts)/2))])
        orgs.append(os)
        parts = line3.split("|")
        #org_scores.append((float(p) for p in parts))
        density.append(float(parts[0]))
        nonuniformity.append(float(parts[1]))
        entropy.append(float(parts[2]))
        intradimoverlap.append(float(parts[3]))

def plot_orgs():
    djs_inx = np.argsort(np.asarray(intradimoverlap))
    djs = ((np.asarray(intradimoverlap)-min(intradimoverlap))/(max(intradimoverlap)-min(intradimoverlap)))[djs_inx]
    dens = ((np.asarray(density)-min(density))/(max(density)-min(density)))[djs_inx]
    #unifs = ((np.asarray(nonuniformity)-min(nonuniformity))/(max(nonuniformity)-min(nonuniformity)))[djs_inx]
    ents = ((np.asarray(entropy)-min(entropy))/(max(entropy)-min(entropy)))[djs_inx]
    plt.plot(djs, label="intradim-overlap")
    plt.plot(dens, label="density")
    #plt.plot(unifs, label="uniformity")
    plt.plot(ents, label="uniformity")
    plt.legend(ncol=1, loc="best", bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.savefig("orgs.pdf")

def print_orgs():
    djs_inx = np.argsort(np.asarray(intradimoverlap))
    print(np.asarray(intradimoverlap)[djs_inx])
    sorted_orgs = np.asarray(orgs)[djs_inx]
    sorted_density = np.asarray(density)[djs_inx]
    sorted_intradimoverlap = np.asarray(intradimoverlap)[djs_inx]
    sorted_entropy = np.asarray(entropy)[djs_inx]
    for i in range(5):
        print('org ' + str(i))
        org = sorted_orgs[i]
        print('dim1')
        print('----')
        print('   '.join([t[1] for t in org[0]]))
        print('dim2')
        print('----')
        print('   '.join([t[1] for t in org[1]]))
        print('------------------------------------------------------')
        print("density: %d - intradim overlap: %d - unformity: %f" % (sorted_density[i], sorted_intradimoverlap[i], sorted_entropy[i]))
        print('------------------------------------------------------')
    for i in range(5):
        print('org ' + str(20000+i))
        org = sorted_orgs[20000+i]
        print('dim1')
        print('----')
        print('   '.join([t[1] for t in org[0]]))
        print('dim2')
        print('----')
        print('   '.join([t[1] for t in org[1]]))
        print('------------------------------------------------------')
        print("density: %d - intradim overlap: %d - unformity: %f" % (sorted_density[20000+i], sorted_intradimoverlap[20000+i], sorted_entropy[20000+i]))
        print('------------------------------------------------------')
    for i in range(5):
        print('org ' + str(500000+i))
        org = sorted_orgs[500000+i]
        print('dim1')
        print('----')
        print('   '.join([t[1] for t in org[0]]))
        print('dim2')
        print('----')
        print('   '.join([t[1] for t in org[1]]))
        print('------------------------------------------------------')
        print("density: %d - intradim overlap: %d - unformity: %f" % (sorted_density[500000+i], sorted_intradimoverlap[500000+i], sorted_entropy[500000+i]))
        print('------------------------------------------------------')



good_labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/good_labels_20k.json', 'r'))
label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_20k.json', 'r'))
print('labels: ')
print(' - '.join([label_names[str(good_labels[i])] for i in range(20)]))

read_organizations(ORG_FILE)
plot_orgs()
print_orgs()

