import org
import org.load as orgl
import org.cluster as orgc
import org.graph as orgg
import org.experiment as orge
import org.hierarchy as orgh
import org.fix as orgf
import importlib
import json
import org.plot.dist_plot as orgp
import datetime
import copy
import operator
import numpy as np


def reload():
    importlib.reload(org)
    importlib.reload(orgl)
    importlib.reload(orgc)
    importlib.reload(orgg)
    importlib.reload(orgh)
    importlib.reload(orge)
    importlib.reload(orgf)

domains = []
keys = []
vecs = []
tagdomains = {}


def init(tag_num):
    global keys, vecs, domains, tagdomains
    print("Loading domains")
    adomains = list(orgl.add_ft_vectors(orgl.iter_domains()))
    domains = domains
    print("Reduce tags")
    atags, atagdomains = orgl.reduce_tag_vectors(adomains)
    print('number of tags: %d' % len(atags))
    atagdomcounts = {t:len(ds) for t, ds in atagdomains.items()}
    satagdomcounts = sorted(atagdomcounts.items(), key=operator.itemgetter(1), reverse=True)
    tvs = [p[0] for p in satagdomcounts]
    tvs = tvs[:tag_num]
    tags = dict()
    tagdomains = dict()
    dcount = 0
    domains = []
    for t in tvs:
        tags[t] = copy.deepcopy(atags[t])
        tagdomains[t] = copy.deepcopy(atagdomains[t])
        dcount += len(tagdomains[t])
        domains.extend(tagdomains[t])
    print('selected %d number of tags and %d datasets.' % (len(tags), dcount))
    keys, vecs = orgc.mk_tag_table(tags)
    return keys, vecs, tagdomains


def kmeans(suffix):
    print('kmeans')
    g = orgc.kmeans_clustering(keys, vecs, 20)
    results = orgh.evaluate(g, domains, tagdomains)
    json.dump(results['success_probs'], open('output/kmeans_tag_dists_' + str(len(domains)) + suffix + '.json', 'w'))
    #json.dump(results['tag_ranks'], open('output/kmeans_tag_ranks_' + str(len(domains)) + suffix + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/kmeans_tag_dists_' + str(len(domains)) + suffix + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/kmeans_prob_hist_' + str(len(domains)) + suffix + '.pdf', 'KMeans Clustering (branching factor=5)')
    print('printed the initial hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/kmeans_prob_hist_' + str(len(domains)) + suffix + '.pdf'))
    return g


def agg(suffix):
    print('agg')
    #params = {'num_clusters': [2], 'measures':[('ward', 'euclidean')]}
    #results = orge.get_tag_ranks_basic(keys, vecs, params, domains)
    s = datetime.datetime.now()
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs)
    e = datetime.datetime.now()
    elapsed = e - s
    print('elapsed time of clustering %d' % int(elapsed.total_seconds() * 1000))
    orgh.init(gp, domains, tagdomains)
    results = orgh.evaluate(gp, domains, tagdomains)

    json.dump(results['success_probs'], open('output/agg_dists' + str(len(domains)) + suffix + '.json', 'w'))
    json.dump(results['tag_ranks'], open('output/agg_ranks' + str(len(domains)) + suffix + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/agg_dists' + str(len(domains)) + suffix + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix + '.pdf', 'Agglomerative Clustering')
    print('printed the initial hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix + '.pdf'))
    return gp, results['success_probs']


def flat(suffix):
    print('flat')
    g = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs)
    orgh.init(g, domains, tagdomains)
    results = orgh.evaluate(g, domains, tagdomains)
    tag_dists = results['success_probs']
    tag_ranks = results['tag_ranks']
    json.dump(tag_dists, open('output/flat_tag_dists_' + str(len(domains)) + suffix + '.json', 'w'))
    json.dump(tag_ranks, open('output/flat_tag_ranks_' + str(len(domains)) + suffix + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/flat_tag_dists_' + str(len(domains)) + suffix + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/flat_prob_' + str(len(domains)) + suffix + '.pdf', 'Flat Hierarchy')
    print('printed to %s' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/flat_prob_' +  str(len(domains)) + suffix + '.pdf'))


def balanced():
    print('balanced')
    g = orgc.complete_kary_cluster(keys, vecs, 5)
    #g = orgc.balanced_clustering(keys, vecs, 5)
    results = orgh.evaluate(g, domains, tagdomains)
    tag_dists = results['success_probs']
    tag_ranks = results['tag_ranks']
    json.dump(tag_dists, open('output/balanced_dists.json', 'w'))
    json.dump(tag_ranks, open('output/balanced_ranks.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/balanced_dists.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/balanced_prob_hist.pdf', 'Balanced Hierarchical Clustering')


def twolevel():
    print('twolevel')
    g = orgc.twolevel_kmeans_clustering(keys, vecs, 5)
    results = orgh.evaluate(g, domains, tagdomains)
    tag_dists = results['success_probs']
    tag_ranks = results['tag_ranks']
    json.dump(tag_dists, open('output/twolevel_tag_dists_max.json', 'w'))
    json.dump(tag_ranks, open('output/twolevel_tag_ranks_max.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/twolevel_tag_dists_max.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/twolevel_reachability_prob_hist.pdf', 'Two-level Hierarchical Clustering')


def fix(g, hierarchy_name):
    print('fix')
    init_results = orgh.evaluate(g.copy(), domains, tagdomains)
    init_success_probs = init_results['success_probs']
    s = datetime.datetime.now()
    h, iteration_sps, iteration_ls = orgf.fix_plus(g, domains, tagdomains)
    print('after fix: node %d edge %d' % (len(h.nodes), len(h.edges)))
    e = datetime.datetime.now()
    elapsed = e - s
    print('elapsed time: %d' % int(elapsed.total_seconds() * 1000))

    results = orgh.evaluate(h, domains, tagdomains)
    success_probs = results['success_probs']
    print('improvement after fixing: %f' % orgh.get_improvement(init_success_probs, success_probs))

    tag_dists = results['success_probs']
    json.dump(tag_dists, open('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json', 'w'))
    json.dump(iteration_sps, open('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json', 'w'))
    json.dump(iteration_ls, open('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json', 'w'))

    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf', 'Fixed Organization')
    orgp.plot_fix('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf', 'Average Success Prob', hierarchy_name)
    orgp.plot_fix('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf', 'Log Likelihood', hierarchy_name)

    print('printed to %s' % ('fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'))
    print('printed iteration success plot to %s' % ('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'))
    print('printed iteration likelihood plot to %s' % ('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'))

    return success_probs



def multidimensional_hierarchy(dim_num):
    global keys, vecs, domains, tagdomains
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    allkeys = list(keys)
    allvecs = copy.deepcopy(vecs)
    alltagdomains = copy.deepcopy(tagdomains)
    hs = dict()
    hscounts = dict()
    success_probs_before = dict()
    success_probs_after = dict()
    for i, ts in dims.items():
        keys = ts
        vecs = []
        domains = []
        tagdomains = dict()
        for t in ts:
            vecs.append(allvecs[allkeys.index(t)])
            domains.extend(alltagdomains[t])
            tagdomains[t] = list(alltagdomains[t])
        vecs = np.array(vecs)
        for d in domains:
            tbl = d['name'][:d['name'].rfind('_')]
            if tbl not in hs:
                hscounts[tbl] = 0
                hs[tbl] = []
            if i not in hs[tbl]:
                hs[tbl].append(i)
                hscounts[tbl] += 1

        print('tags: %d %d' % (len(keys), len(vecs)))
        print('domains: %d  tagdomains: %d' % (len(domains), len(tagdomains)))
        print('fixing cluster %d' % i)
        gp, sps = agg('agg'+str(i))
        for t, p in sps.items():
            if t not in success_probs_before:
                success_probs_before[t] = 0.0
            success_probs_before[t] += p

        sps = fix(gp, 'agg_'+str(i))
        for t, p in sps.items():
            if t not in success_probs_after:
                success_probs_after[t] = 0.0
            success_probs_after[t] += p
        print('---------------')
    print(hscounts)
    sp = sum(list(success_probs_before.values()))/float(len(success_probs_before))
    print('success prob of multidimensions before fix: %f' % sp)
    sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of multidimensions after fix: %f' % sp)



init(100)#364)

#kmeans('')
#agg('')
flat('flat')
#balanced()
#twolevel()

start = datetime.datetime.now()
print(start)
#fix(kmeans(''), 'kmeans')
#h, res = agg('bottomup_reduce_height')
#fix(h, 'agg_bottomup_reduce_height')
print('agglomerative')
agg('')
#multidimensional_hierarchy(2)

end = datetime.datetime.now()
print(end)
elapsed = end - start
print('elapsed time: %d' % int(elapsed.total_seconds() * 1000))



print("Done")


