import json
import numpy as np
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def all_pair_sim(domains, simsfile):
    print('all_pair_sim for %d domains' % len(domains))
    s = datetime.datetime.now()
    # computing all pair sim between domains
    # potentially using lsh for scalability
    sims = dict()
    for i in range(len(domains)):
        d1 = domains[i]
        if ((i+1)%100) == 0:
            print('processed pairs for %d domains.' % i)
            print('elapsed time of pair sim calc %d' % int((datetime.datetime.now()-s).total_seconds() * 1000))

        for j in range(i, len(domains)):
            d2 = domains[j]
            sim = 1.0
            if i != j:
                sim = get_sim(d1['mean'], d2['mean'])
            if sim < 0.5:
                continue
            if d1['name'] not in sims:
                sims[d1['name']] = dict()
            sims[d1['name']][d2['name']] = sim
            if d2['name'] not in sims:
                sims[d2['name']] = dict()
            sims[d2['name']][d1['name']] = sim
    e = datetime.datetime.now()
    elapsed = e - s
    print('elapsed time of all pair sim calc %d' % int(elapsed.total_seconds() * 1000))
    print('all_sim: %d domains: %d' % (len(sims), len(domains)))
    json.dump(sims, open(simsfile, 'w'))
    print('done all_pair_sim')



def make_cloud(simfile, threshold):
    print('make_cloud')
    sims = json.load(open(simfile, 'r'))
    aps = 0
    pts = 0
    mincloudsize = len(sims)
    maxcloudsize = 0
    # apply threshold on the sim of domains to a target
    cloud = dict()
    for d1, d2sims in sims.items():
        cloud[d1] = dict()
        for d2, s in d2sims.items():
            aps += 1
            if s > threshold:
                pts += 1
                cloud[d1][d2] = s
        maxcloudsize = max(maxcloudsize, len(cloud[d1]))
        mincloudsize = min(mincloudsize, len(cloud[d1]))

    print("%d out of %d pairs passed the threshold." % (pts, aps))
    print("size of clouds vary from %d to %d" % (mincloudsize, maxcloudsize))
    return cloud


def plot(cloud):
    pdffile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/cloud.pdf'
    ys = [len(ds) for t, ds in cloud.items()]
    print('len(ys): %d' % len(ys))
    print('>2: %d' % sum(i > 1 for i in ys))
    print('min clo: %d' % min(ys))
    print('max clo: %d' % max(ys))
    print('avg clo: %d' % (sum(ys)/float(len(ys))))
    ys.sort(reverse=True)
    print('number of accepted doms: min:  %d  max:  %d' % (min(ys), max(ys)))
    print('cloud size > 1: %d' % len([i for i in ys if i>1]))
    plt.plot([i for i in range(len(ys))], ys)
    plt.xlabel('target domain')
    plt.ylabel('cloud size')
    plt.title('accepted domains for each target')
    plt.savefig(pdffile)
    plt.clf()
    print('ploted cloud stats to %s' % pdffile)


def get_sim(vec1, vec2):
    c = float(max(0.000001, cosine(vec1, vec2)))
    return c


def cosine(vec1, vec2):
    dp = np.dot(vec1,vec2)
    c = dp / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return c




