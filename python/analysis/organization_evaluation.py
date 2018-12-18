import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("acm-2col.mplstyle")
import json

ORGS_EVALUATION_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/orgs_evaluation_10.json'

org_evals = json.load(open(ORGS_EVALUATION_FILE, 'r'))

inx = np.argsort(np.asarray(org_evals))[::-1]
ys = np.asarray(org_evals)[inx]
xs = [i for i in range(len(ys))]
fs = 8
lw = 2
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Organization', fontsize=fs)
ax.set_ylabel('Expected Success Probability', fontsize=fs)
ax.set_xlim([0,len(xs)])
ax.set_ylim([0, max(org_evals)])
ax.set_axisbelow(True)
ax.plot(xs, ys, "-", color = 'royalblue', linewidth=lw)
plt.savefig("plots/org_evals_10.pdf",  bbox_inches='tight', pad_inches=0.02)
plt.close()


