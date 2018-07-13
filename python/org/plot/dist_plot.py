import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reachability prob histogram
results = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/agg_tag_dists_2.json', 'r'))
ps = []
for t, ds in results.items():
    for d in ds:
        if d[0] == t:
            ps.append(d[1])
weights = np.ones_like(ps)/float(len(ps))
plt.hist(ps, bins=list(np.linspace(0.0,1.0,100)), weights=weights)
plt.xlabel('Domain Reachability Probability')
plt.title('K-means Hierarchical Clustering (branching factor = 5)')
plt.xlim([0.0,1.05])
plt.xticks(np.arange(0.0, 1.05, 0.1))
plt.savefig('agg_reachability_prob_hist_2.pdf')


