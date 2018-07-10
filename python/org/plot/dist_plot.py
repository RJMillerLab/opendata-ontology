import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reachability prob histogram
results = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/kmeans_tag_dists.json', 'r'))
ps = []
for t, ds in results.items():
    for d in ds:
        if d[0] == t:
            ps.append(d[1])
plt.hist(ps, bins=list(np.linspace(0.0,1.0,100)))
plt.xlabel('Probability')
plt.title('Histogram of Domain Reachability Prob.')
plt.xlim([0.0,1.05])
plt.xticks(np.arange(0.0, 1.05, 0.1))
plt.savefig('kmeans_reachability_prob_hist.pdf')


