import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/tag_ranks.json', 'r'))
tag_ranks = list(results.values())
yds = {i:tag_ranks.count(i) for i in tag_ranks}
ys = [0 for i in range(1,max(list(yds.keys()))+1)]
for k, v in yds.items():
    ys[k-1] = v
xs = [i for i in range(1,max(list(yds.keys()))+1)]
plt.bar(xs, ys)
plt.xlabel('Rank')
plt.ylabel('Number of Tags')
plt.title('Ranking 500 of Tags based on Reachability Prob.')
plt.xlim([1,max(tag_ranks)])
plt.xticks(np.arange(0, len(xs)+2, 10))
plt.savefig('tag_rank_hist.pdf')

# reachability prob histogram
results = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/kmeans_tag_ranks.json', 'r'))
ys = []
for v in list(results.values()):
    ys.append(v)
plt.hist(ys, bins=list(np.linspace(0,max(ys),max(ys))))
plt.xlabel('Rank')
plt.title('Histogram of Domain Reachability Ranks.')
plt.xlim([0,max(ys)])
plt.xticks(np.arange(0, max(ys)+1, 10))
plt.savefig('kmeans_rank_hist.pdf')


