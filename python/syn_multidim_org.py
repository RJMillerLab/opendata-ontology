import org.hierarchy as orgh
import org.graph as orgg
import org.plot.dist_plot as orgp
import org.cluster as orgc
import org.cloud as orgk
import org.load as orgl
import org.fix as orgf
import operator
import numpy as np
import json
import copy
import datetime


keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
domainclouds = dict()


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
    nudomains = []
    for t in tvs:
        tags[t] = copy.deepcopy(atags[t])
        tagdomains[t] = copy.deepcopy(atagdomains[t])
        nudomains.extend(tagdomains[t])
    domains = []
    # making domains unique
    seen = dict()
    for domain in nudomains:
        if domain['name'] not in seen:
            domains.append(domain)
            seen[domain['name']] = True
    print('domains: %d  -> domains: %d' % (len(nudomains), len(domains)))
    keys, vecs = orgc.mk_tag_table(tags)
    return keys, vecs, tagdomains




def init_plus():
    global domainclouds
    # finding the cloud
    simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
    domainclouds = orgk.make_cloud(simfile, 0.75)
    orgk.plot(domainclouds)



def fix(g, hierarchy_name):
    print('fix')
    h, iteration_sps, iteration_ls = orgf.fix_plus(g, domains, tagdomains, domainclouds, 'synthetic')

    print('fuzzy: ')
    results = orgh.fuzzy_evaluate(h.copy(), domains, tagdomains, domainclouds, 'synthetic')
    success_probs = results['success_probs']

    json.dump(success_probs, open('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json', 'w'))
    json.dump(iteration_sps, open('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json', 'w'))
    json.dump(iteration_ls, open('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json', 'w'))

    fix_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json'
    fix_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'
    orgp.plot(fix_json, fix_pdf, 'Fixed Organization')

    sps_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json'
    sps_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'
    orgp.plot_fix(sps_json, sps_pdf, 'Average Success Prob', hierarchy_name)

    ls_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json'
    ls_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'
    orgp.plot_fix(ls_json, ls_pdf, 'Log Likelihood', hierarchy_name)

    print('printed to %s' % ('fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'))
    print('printed iteration success plot to %s' % ('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'))
    print('printed iteration likelihood plot to %s' % ('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'))

    return success_probs



def agg_fuzzy(suffix1, suffix2):
    print('agg_fuzzy')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs)
    orgh.init(gp, domains, tagdomains)
    results = orgh.fuzzy_evaluate(gp.copy(), domains, tagdomains, domainclouds, 'synthetic')
    success_probs = results['success_probs']
    avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))

    print("ploting")
    json.dump(results['success_probs'], open('synthetic_output/agg_dists' + str(len(domains)) + suffix1 + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists' + str(len(domains)) + suffix1 + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf', 'simbased eval - single dimension - ' + str(avg_success_prob))
    print('printed the fuzzy eval of agg hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf'))

    return gp, results['success_probs']

    print("evaluating")
    results = orgh.fuzzy_evaluate(gp.copy(), domains, tagdomains, domainclouds, 'synthetic')
    success_probs = results['success_probs']
    avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))
    print("ploting")
    json.dump(results['success_probs'], open('synthetic_output/agg_dists' + str(len(domains)) + suffix2 + '.json', 'w'))
    json.dump(results['tag_ranks'], open('synthetic_output/agg_ranks' + str(len(domains)) + suffix2 + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists' + str(len(domains)) + suffix2 + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix2 + '.pdf', 'strict eval - single dimension - ' + str(avg_success_prob))
    print('printed the initial hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix2 + '.pdf'))



def multidimensional_hierarchy(dim_num):
    print('multidimensional_hierarchy')
    global keys, vecs, domains, tagdomains, domainclouds
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    allkeys = list(keys)
    allvecs = copy.deepcopy(vecs)
    alltagdomains = copy.deepcopy(tagdomains)
    alldomains = copy.deepcopy(domains)
    alldomainclouds = copy.deepcopy(domainclouds)
    success_probs_before = dict()
    success_probs_before_intersect = dict()
    success_probs_after = dict()
    success_probs_after_intersect = dict()
    for i, ts in dims.items():
        print('dim %d' % i)
        ds = datetime.datetime.now()
        keys = ts
        vecs = []
        domains = []
        domainclouds = dict()
        domain_names = {}
        tagdomains = dict()
        for t in ts:
            vecs.append(allvecs[allkeys.index(t)])
            for td in alltagdomains[t]:
                if td['name'] not in domain_names:
                    domains.append(td)
                    domain_names[td['name']] = True
            tagdomains[t] = list(alltagdomains[t])
        vecs = np.array(vecs)
        for d1 in domain_names:
            domainclouds[d1] = dict()
            for d2, s in alldomainclouds[d1].items():
                if d2 in domain_names:
                    domainclouds[d1][d2] = s
        print('tags: %d vecs: %d domains: %d  tagdomains: %d  domainclouds: %d' % (len(keys), len(vecs), len(domains), len(tagdomains), len(domainclouds)))

        gp, sps = agg_fuzzy('agg'+str(i)+'f2op', '')
        for t, p in sps.items():
            if t not in success_probs_before:
                success_probs_before[t] = 0.0
                success_probs_before_intersect[t] = 1.0
            success_probs_before[t] += p
            success_probs_before_intersect[t] *= p


        de = datetime.datetime.now()
        delapsed = de - ds
        print('elapsed time of dim %d is %d' % (i, int(delapsed.total_seconds() * 1000)))

        print('fixing cluster %d' % i)
        sps = fix(gp, 'agg_'+str(i)+'f2op')
        sp = sum(list(sps.values()))/float(len(sps))
        print('sp of dim %d after fix is %f.' % (i, sp))
        for t, p in sps.items():
            if t not in success_probs_after:
                success_probs_after[t] = 0.0
                success_probs_after_intersect[t] = 1.0
            success_probs_after[t] += p
            success_probs_after_intersect[t] *= p
        print('---------------')
    for t, p in success_probs_before.items():
        if success_probs_before[t] != success_probs_before_intersect[t]:
            success_probs_before[t] = (p-success_probs_before_intersect[t])
        if success_probs_before[t] > 1.0:
            success_probs_before[t] = 1.0
    for t, p in success_probs_after.items():
        if success_probs_after[t] != success_probs_after_intersect[t]:
            success_probs_after[t] = (p-success_probs_after_intersect[t])
        if success_probs_after[t] > 1.0:
            success_probs_after[t] = 1.0

    before_sp = sum(list(success_probs_before.values()))/float(len(success_probs_before))
    print('success prob of multidimensions before fix: %f' % before_sp)

    after_sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of multidimensions after fix: %f' % after_sp)

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_before_' + str(len(alldomains)) + '_' + str(dim_num) + '.json'
    json.dump(success_probs_before, open(multi_json, 'w'))
    print('printed mult before results to %s.' % multi_json)

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_' + str(len(alldomains)) + '_' + str(dim_num) + '_g10rhap.json'
    json.dump(success_probs_after, open(multi_json, 'w'))
    print('printed mult after results to %s.' % multi_json)



def multidim():
    global keys, vecs
    dims = orgc.cmeans_clustering(keys, vecs)
    print('vecs: %d' % len(vecs))
    for c, d in dims.items():
        print(len(d['tags']))


init(500)

#simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
#orgk.all_pair_sim(domains, simfile)

init_plus()

multidimensional_hierarchy(2)

#multidim()

