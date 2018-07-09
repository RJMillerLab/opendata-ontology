import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reachability prob histogram
results = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/kmeans_tag_dists.json', 'r'))
ys = []
for t, ds in results.items():
    tprob = 0.0
    ps = []
    for d in ds:
        if d[0] == t:
            tprob = d[1]
        ps.append(d[1])
    ps.sort()
    ys.append(ps.index(tprob)/float(len(ps)))
ys.sort()
plt.plot([i for i in range(len(ys))], ys)
plt.xlabel('domain tag')
plt.ylabel('goodness')
plt.savefig('tag_prob_goodness.pdf')


