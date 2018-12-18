import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("acm-2col.mplstyle")
import json

TABLE_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'
TABLE_BOOSTED_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/tables_31k.boosted_labels'
MODEL_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/labels_20k.models'
LABEL_NAMES = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'
tls = json.load(open(TABLE_LABELS_FILE, 'r'))
boostedtls = json.load(open(TABLE_BOOSTED_LABELS_FILE, 'r'))
labels = json.load(open(MODEL_LABELS_FILE, 'r')).keys()
label_names = json.load(open(LABEL_NAMES, 'r'))

tlfs = {}
boostedtlfs = {}
ltfs = {}
boostedltfs = {}
y1s = [] # tag/label dist before boosting
y2s = [] # tag/label dist after boosting
y3s = [] # label/tag dist before boosting
y4s = [] # label/tag dist before boosting
for t, ls in tls.items():
    tlfs[t] = len(ls)
    for l in ls:
        if str(l) not in labels:
            continue
        if str(l) not in ltfs:
            ltfs[str(l)] = 1
        else:
            ltfs[str(l)] += 1
for t, ls in boostedtls.items():
    boostedtlfs[t] = len(ls)
    for l, p in ls.items():
        if str(l) not in labels:
            continue
        if str(l) not in boostedltfs:
            boostedltfs[str(l)] = 1
        else:
            boostedltfs[str(l)] += 1
diffs = []
for t, f in tlfs.items():
    y1s.append(f)
    y2s.append(boostedtlfs[t])
    diffs.append(boostedtlfs[t]-f)
print(sum(diffs))
diffs2 = []
ls = []
for l, f in ltfs.items():
    if l not in labels:
        continue
    y3s.append(f)
    y4s.append(boostedltfs[l])
    diffs2.append(boostedltfs[l]-f)
    ls.append(l)
sdiffs2 = diffs2
inx = np.argsort(np.asarray(sdiffs2))[::-1]
effective_tag_ids = [i for i in inx if np.asarray(sdiffs2)[i]>0]
effective_tags = [label_names[ls[i]] for i in effective_tag_ids]
#print('number of effective tags: %d' % len(effective_tag_ids))
#print('effective tags: ')
#print(effective_tags)
#print(sum(diffs2))
#print("len(y3s): %d" % len(y3s))
#print("len(y1s): %d" % len(y1s))
inx = np.argsort(np.asarray(y1s))[::-1]
y1s = np.asarray(y1s)[inx]
y2s = np.asarray(y2s)[inx]
diffs = np.asarray(diffs)[inx]
xs = [i for i in range(len(y1s))]
fs = 8
lw = 2
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Dataset', fontsize=fs)
ax.set_ylabel('Number of Tags', fontsize=fs)
ax.set_xlim([0,len(xs)])
ax.set_ylim([0, max(max(y1s), max(y2s))+1])
ax.set_title('Tags Per Dataset Boost by Classifiers', fontsize=fs)
ax.set_axisbelow(True)
ax.plot(xs, y1s, "-", color = 'royalblue', alpha=0.7, linewidth=lw,  label='before boosting')
ax.plot(xs, y2s, "-", color = 'red', linewidth=lw,alpha=0.7,linestyle='--', label='after boosting')#alpha=0.7,
lgd = plt.legend(ncol=1, loc="best", bbox_transform=plt.gcf().transFigure,          fontsize=fs, fancybox=True, framealpha=0.5)
plt.savefig("plots/table_labels.pdf", bbox_extra_artists=(lgd,),               bbox_inches='tight', pad_inches=0.02)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Tag', fontsize=fs)
ax.set_ylabel('Number of Datasets', fontsize=fs)
ax.set_title('Datasets Found for Tags by Classifiers', fontsize=fs)
xs = [i for i in range(len(diffs2))]
ax.set_xlim([0,len(xs)])
ax.set_ylim([0, max(diffs2)+50])
ax.set_axisbelow(True)
ax.plot(xs, diffs2, "-", color = 'blue', linewidth=lw, markevery=5)
plt.tight_layout()
plt.savefig("plots/diff_label_tables.pdf")
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Dataset', fontsize=fs)
ax.set_ylabel('Number of Tags', fontsize=fs)
ax.set_title('Tags Added to Datasets by Classifiers', fontsize=fs)
xs = [i for i in range(len(diffs))]
ax.set_xlim([0,len(xs)])
ax.set_ylim([0, max(diffs)+2])
ax.set_axisbelow(True)
ax.plot(xs, diffs, "-", color = 'green', linewidth=lw, markevery=5)
plt.tight_layout()
plt.savefig("plots/diff_table_labels.pdf")
plt.close()

plt.hist(diffs2, bins=list(np.linspace(0.0,len(set(diffs2)),len(set(diffs2)))))
plt.xlabel('Tags Added to Datasets')
plt.xlim([0,len(set(diffs2))+1])
plt.xticks(np.arange(0,len(set(diffs2))+1,1))
plt.savefig('plots/diff_table_labels_hist.pdf')

inx = np.argsort(np.asarray(y3s))[::-1]
y3s = np.asarray(y3s)[inx]
y4s = np.asarray(y4s)[inx]
xs = [i for i in range(len(y3s))]
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Tag', fontsize=fs)
ax.set_ylabel('Number of Datasets', fontsize=fs)
ax.set_title('Datasets Per Tag Boost by Classifiers', fontsize=fs)
ax.set_xlim([0,len(xs)])
ax.set_ylim([0, max(max(y3s), max(y4s))+1])
ax.set_axisbelow(True)
ax.plot(xs, y3s, "-", color = 'black', linewidth=lw,linestyle='--', alpha=0.7, label='before boosting')
ax.plot(xs, y4s, "-", color = 'orange', linewidth=lw, alpha=0.7, label='after boosting')
lgd = plt.legend(ncol=1, loc="best", bbox_transform=plt.gcf().transFigure,          fontsize=fs, fancybox=True, framealpha=0.5)
plt.savefig("plots/label_tables.pdf", bbox_extra_artists=(lgd,),               bbox_inches='tight', pad_inches=0.02)
plt.close()


