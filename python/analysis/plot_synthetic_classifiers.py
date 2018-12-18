import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("acm-2col.mplstyle")
import json

CLASSIFICATION_RESULTS = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/model_results.json'
results = json.load(open(CLASSIFICATION_RESULTS, 'r'))

precisions = []
recalls = []
accuracies = []
f1s = []

for t, rs in results.items():
    precisions.append(rs['precision'])
    recalls.append(rs['recall'])
    accuracies.append(rs['accuracy'])
    f1s.append(rs['f1'])

inx = np.argsort(np.asarray(precisions))[::-1]

xs = [i for i in range(len(results))]
y1s = np.asarray(precisions)[inx]
y2s = np.asarray(recalls)[inx]
y3s = np.asarray(accuracies)[inx]
y4s = np.asarray(f1s)[inx]

fs = 8
lw = 2
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Tag', fontsize=fs)
ax.set_ylabel('Measure', fontsize=fs)
ax.set_xlim([0,len(xs)])
ax.set_ylim([0.6,1.0])
ax.set_title('Classifiers on Synthetic Data', fontsize=fs)
ax.set_axisbelow(True)
ax.plot(xs, y1s, "-", color = 'royalblue', alpha=0.7, linewidth=lw,  label='precision')
ax.plot(xs, y2s, "-", color = 'red', linewidth=lw,alpha=0.7,linestyle='--', label='recall')
#ax.plot(xs, y2s, "-", color = 'black', linewidth=lw,alpha=0.7,linestyle=':', label='accuracy')
ax.plot(xs, y2s, "-", color = 'green', linewidth=lw,alpha=0.7,linestyle='-.', label='f1')
ax.yaxis.grid(linestyle="dotted")
ax.xaxis.grid(linestyle="dotted")
lgd = plt.legend(ncol=1, loc="best", bbox_transform=plt.gcf().transFigure, fontsize=fs, fancybox=True, framealpha=0.5)
plt.savefig("plots/synthetic_classifiers.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.02)
plt.close()

