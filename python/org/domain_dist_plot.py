import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'navy', 'peru', 'teal']

# reachability prob histogram
results = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/kmeans_tag_dists.json', 'r'))
doms = []
for t, ds in results.items():
    if len(doms) > 5:
        continue
    doms.append([p[1] for p in ds])
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
ax1.hist(doms[0], bins=np.linspace(0.0,1.0,100))
ax1.set_title('domain 1')
ax2.hist(doms[1], bins=np.linspace(0.0,1.0,100))
ax2.set_title('domain 2')
ax3.hist(doms[2], bins=np.linspace(0.0,1.0,100))
ax3.set_title('domain 3')
ax4.hist(doms[3], bins=np.linspace(0.0,1.0,100))
ax4.set_title('domain 4')
ax5.hist(doms[4], bins=np.linspace(0.0,1.0,100))
ax5.set_title('domain 5')
ax5.set_xlabel('Reachability Probability')
ax6.hist(doms[5], bins=np.linspace(0.0,1.0,100))
ax6.set_title('domain 6')
plt.savefig('kmeans_domain_reachability_hist.pdf')


