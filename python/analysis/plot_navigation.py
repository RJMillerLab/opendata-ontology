import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("acm-2col.mplstyle")
import json

TRANSITION_PROB_FILE = "/home/fnargesian/FINDOPENDATA_DATASETS/10k/transition_prob.json"
STATE_PROB_FILE = "/home/fnargesian/FINDOPENDATA_DATASETS/10k/state_prob.json"
TABLE_LABEL_SELECTIVITY = "/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_label_selectivity.json"

trans_probs = json.load(open(TRANSITION_PROB_FILE, 'r'))
state_probs = json.load(open(STATE_PROB_FILE, 'r'))
table_label_selects = json.load(open(TABLE_LABEL_SELECTIVITY, 'r'))

# plotting transition probabilities
pairs = []
for s1, s2m in trans_probs.items():
    for s2, p in s2m.items():
        pairs.append(s1+" -> "+s2)
ys = [p for s1, s2m in trans_probs.items() for s2, p in s2m.items()]
ys.sort()
xs = [i for i in range(len(ys))]
fs = 8
lw = 2
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('State Pair', fontsize=fs)
ax.set_ylabel('Transition Probability', fontsize=fs)
ax.set_ylim([0, max(ys)+0.05])
ax.set_axisbelow(True)
ax.plot(xs, ys, "-", color = 'royalblue', linewidth=lw, markevery=5)
plt.tight_layout()
plt.savefig("plots/transition_prob.pdf")
plt.close()
# printing transition probs
ys = [p for s1, s2m in trans_probs.items() for s2, p in s2m.items()]
s_ys_inx = np.argsort(ys)
print(np.asarray(pairs)[s_ys_inx[::-1]][:20])
print(np.asarray(ys)[s_ys_inx[::-1]][:20])
# plotting state probabilities
ys = [p for s, p in state_probs.items()]
ys.sort()
xs = [i for i in range(len(ys))]
fs = 11
lw = 2
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('State', fontsize=fs)
ax.set_ylabel('Complexity', fontsize=fs)
#ax.set_xlim([1,len(ys)])
ax.set_ylim([0, max(ys)+0.1])
ax.set_axisbelow(True)
ax.plot(xs, ys, "-", color = 'darkgreen', linewidth=lw, markevery=5)
plt.tight_layout()
plt.savefig("plots/state_complexity.pdf")
plt.close()

#
table_avg_selects = [sum(ss)/float(len(ss)) for t, ss in table_label_selects.items()]
print("average number of selectable states across tables: %f" % (sum(table_avg_selects)/float(len(table_avg_selects))))
ys = table_avg_selects
ys.sort()
xs = [i for i in range(len(ys))]
fs = 11
lw = 2
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.set_xlabel('Table', fontsize=fs)
ax.set_ylabel('Average Selectability', fontsize=fs)
ax.set_ylim([0, max(ys)+1])
ax.set_axisbelow(True)
ax.plot(xs, ys, "-", color = 'red', linewidth=lw, markevery=5)
plt.tight_layout()
plt.savefig("plots/table_selectable.pdf")
plt.close()



