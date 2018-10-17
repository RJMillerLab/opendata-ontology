import org.od_hierarchy as orgh
import org.graph as orgg
import org.cluster as orgc
import org.load as orgl
import org.od_fix as orgf
import operator
import numpy as np
import json
import copy
import datetime
import org.cloud as orgk



keys = []
vecs = np.array([])
domains = []
repdomains = dict()
tagdomains = dict()
domainclouds = dict()
domaintags = dict()
tagdomsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/rep_domain_sims.json'
simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
reps = []

def get_reps():
    global repdomains, reps
    for i in range(len(keys)):
        reps.append({'name': keys[i], 'mean': vecs[i]})
    repdomains = copy.deepcopy(tagdomains)
    print('repdomains: %d' % len(repdomains))
    print('tagdomains: %d' % len(tagdomains))


def init(tag_num):
    global keys, vecs, domains, tagdomains, domaintags, domainclouds
    print("Loading domains")
    adomains = list(orgl.add_ft_vectors(orgl.iter_domains()))
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

    for dom in adomains:
        if dom['name'] not in domaintags:
            domaintags[dom['name']] = []
        if dom['tag'] not in domaintags[dom['name']]:
            domaintags[dom['name']].append(dom['tag'])
    print('domain tags: %d' % len(domaintags))
    for t in tvs:
        tags[t] = copy.deepcopy(atags[t])
        tagdomains[t] = [dom['name'] for dom in atagdomains[t]]
        nudomains.extend(list(atagdomains[t]))
    domains = []
    # making domains unique
    seen = dict()
    for domain in nudomains:
        if domain['name'] not in seen:
            domains.append(domain)
            seen[domain['name']] = True
    print('domains: %d  -> domains: %d' % (len(nudomains), len(domains)))
    keys, vecs = orgc.mk_tag_table(tags)

    # creating reps
    get_reps()

    # Tag-rep sims are precalculated.
    orgh.get_tag_domain_sim(reps, keys, vecs, tagdomsimfile)

    # finding the cloud
    simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
    domainclouds = orgk.make_cloud(simfile, 0.75)



def init_plus():
    global domainclouds
    # finding the cloud
    simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
    domainclouds = orgk.make_cloud(simfile, 0.75)



def fix(g, hierarchy_name):
    print('fix')
    h, stats, iteration_ls, sps, dsps = orgf.fix_plus(g, domains, tagdomains, domainclouds, 'synthetic', domaintags, reps, repdomains)

    print('fuzzy eval: ')
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_rep_prob_fuzzy(h.copy(), domains, tagdomains, domainclouds, 'synthetic', domaintags, repdomains, reps)

    json.dump(success_probs, open('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json', 'w'))
    json.dump(domain_success_probs, open('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_domain_success_probs.json', 'w'))
    json.dump(iteration_ls, open('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json', 'w'))

    print('printed to %s' % ('fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'))
    print('printed iteration success plot to %s' % ('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'))
    print('printed iteration likelihood plot to %s' % ('synthetic_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'))

    return success_probs, domain_success_probs, stats



def agg_fuzzy(suffix1, suffix2):
    print('agglomerative')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)

    print('domains: %d' % len(domains))

    orgh.get_state_domain_sims(gp, tagdomsimfile, reps)
    orgh.init(gp, domains, simfile)
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_rep_prob_fuzzy(gp.copy(), domains, tagdomains, domainclouds, 'synthetic', domaintags, repdomains, reps)

    print(max_success)
    json.dump(success_probs, open('synthetic_output/agg_dists' + str(len(domains)) + suffix1 + '.json', 'w'))
    print('printed the fuzzy eval of agg hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf'))

    return gp, success_probs, domain_success_probs



def multidimensional_hierarchy(dim_num):
    print('multidimensional_hierarchy')
    global keys, vecs, domains, tagdomains, repdomains, domainclouds
    stats = []
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    allkeys = list(keys)
    allvecs = copy.deepcopy(vecs)
    alltagdomains = copy.deepcopy(tagdomains)
    allrepdomains = copy.deepcopy(repdomains)
    alldomains = copy.deepcopy(domains)
    alldomainclouds = copy.deepcopy(domainclouds)
    success_probs_before = dict()
    success_probs_before_intersect = dict()
    success_probs_after = dict()
    success_probs_after_intersect = dict()
    # build an index of domains on name
    domain_index = dict()
    for i in range(len(domains)):
        if domains[i]['name'] not in domain_index:
            domain_index[domains[i]['name']] = i
    #
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
                if td not in domain_names:
                    domains.append(alldomains[domain_index[td]])
                    domain_names[td] = True
            tagdomains[t] = list(alltagdomains[t])
        vecs = np.array(vecs)
        for d1 in domain_names:
            domainclouds[d1] = dict()
            for d2, s in alldomainclouds[d1].items():
                if d2 in domain_names:
                    domainclouds[d1][d2] = s
        print('tags: %d vecs: %d domains: %d  tagdomains: %d  domainclouds: %d' % (len(keys), len(vecs), len(domains), len(tagdomains), len(domainclouds)))

        for rep, rdoms in allrepdomains.items():
            repdomains[rep] = [rd for rd in rdoms if rd in domain_names]

        gp, sps, before_dsps = agg_fuzzy('agg'+str(i)+'f2op_partial_sim_threshold06_stats', '')
        for t, p in sps.items():
            if t not in success_probs_before:
                success_probs_before[t] = 0.0
                success_probs_before_intersect[t] = 1.0
            success_probs_before[t] += p
            success_probs_before_intersect[t] *= p

        print('fixing cluster %d' % i)
        sps, after_dsps, dim_stats = fix(gp, 'agg_'+str(i)+'f2op_partial_sim_threshold062')
        stats.extend(dim_stats)
        sp = sum(list(sps.values()))/float(len(sps))
        print('sp of dim %d after fix is %f.' % (i, sp))
        for t, p in sps.items():
            if t not in success_probs_after:
                success_probs_after[t] = 0.0
                success_probs_after_intersect[t] = 1.0
            success_probs_after[t] += p
            success_probs_after_intersect[t] *= p
        de = datetime.datetime.now()
        delapsed = de - ds
        print('elapsed time of dim %d is %d' % (i, int(delapsed.total_seconds() * 1000)))
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

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_partial_sim_threshold062_before_' + str(len(alldomains)) + '_' + str(dim_num) + '.json'
    json.dump(success_probs_before, open(multi_json, 'w'))
    print('printed mult before results to %s.' % multi_json)

    multi_dom_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_domain_probs_partial_sim_threshold062_before_' + str(len(alldomains)) + '_' + str(dim_num) + '_g10rhap.json'
    json.dump(before_dsps, open(multi_dom_json, 'w'))
    print('printed domain mult results before to %s.' % multi_dom_json)

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_dists_partial_sim_threshold062_' + str(len(alldomains)) + '_' + str(dim_num) + '_g10rhap.json'
    json.dump(success_probs_after, open(multi_json, 'w'))
    print('printed mult after results to %s.' % multi_json)

    multi_dom_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/multidim_domain_probs_partial_sim_threshold062_after_' + str(len(alldomains)) + '_' + str(dim_num) + '_g10rhap.json'
    json.dump(after_dsps, open(multi_dom_json, 'w'))
    print('printed domain mult results after fix to %s.' % multi_dom_json)

    json.dump(stats, open('synthetic_output/fix_' + str(len(domains)) + '_prunning_stats.json',    'w'))
    print('printed stats to %s' % ('synthetic_output/fix_' + str(len(domains)) + '_prunning_stats.json'))




print('started at: ')
print(datetime.datetime.now())

init(500)

multidimensional_hierarchy(2)

print('ended at:')
print(datetime.datetime.now())

#multidim()
#orgk.all_pair_sim(domains, simfile)
