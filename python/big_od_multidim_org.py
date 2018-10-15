import org.optimized_hierarchy as orgh
import org.graph as orgg
import org.plot.dist_plot as orgp
import org.cluster as orgc
import org.sample as orgs
import org.cloud as orgk
import org.optimized_fix as orgf
import numpy as np
import json
import copy
import datetime

simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json'
dimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_dims.json'
traindomainfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_sample_domain_names.json'
tagdomsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_domain_sims.json'
keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
domainclouds = dict()
domaintags = dict()
testdomains = []
testtagdomains = dict()
testdomainclouds = dict()
testdomaintags = dict()
testdomainsclouds = dict()
dimdomains = []

dim_num = 10


def init_load_dim(dim_id):

    global testdomainclouds, domainclouds, keys, vecs, testdomains, testtagdomains, testdomaintags,  domains, tagdomains, domaintags, dimdomains

    domaintags, tagdomains, testdomaintags, testtagdomains, testdomainclouds, domainclouds = dict(), dict(), dict(), dict(), dict(), dict()
    domains, testdomains = [], []

    dims = json.load(open(dimfile, 'r'))
    dimtags = dims[dim_id]

    sample_domain_names = json.load(open(traindomainfile, 'r'))

    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        if t.startswith('socrata_') and t in dimtags:
            ks.append(t)
            vs.append(e)


    print('tag_embs: %d dim ks: %d vs: %d' % (len(tag_embs), len(vs), len(vs)))

    keys = ks
    vecs = np.array(vs)

    alldomains = json.load(open(DOMAIN_FILE))

    tbs, ttbs = [], []
    traindomainnames = dict()
    testdomainnames = dict()
    atagdomains = dict()
    dimdomains = []
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue

        if dom['tag'] not in atagdomains:
            atagdomains[dom['tag']] = []
        if dom['name'] not in atagdomains[dom['tag']]:
            atagdomains[dom['tag']].append(dom['name'])

        if dom['name'] not in sample_domain_names:
            if dom['name'] not in testdomainnames:
                testdomainnames[dom['name']] = True
                testdomains.append(dom)
                dimdomains.append(dom)

            if dom['name'] not in testdomaintags:
                testdomaintags[dom['name']] = []
            if dom['tag'] not in testdomaintags[dom['name']]:
                testdomaintags[dom['name']].append(dom['tag'])

            table = dom['name'][:dom['name'].rfind('_')]
            if table not in ttbs:
                ttbs.append(table)

            continue


        if dom['name'] not in traindomainnames:
            traindomainnames[dom['name']] = True
            domains.append(dom)
            if dom['name'] not in testdomainnames:
                dimdomains.append(dom)

        if dom['name'] not in domaintags:
            domaintags[dom['name']] = []
        if dom['tag'] not in domaintags[dom['name']]:
            domaintags[dom['name']].append(dom['tag'])

        table = dom['name'][:dom['name'].rfind('_')]
        if table not in tbs:
            tbs.append(table)


        if dom['tag'] not in tagdomains:
            tagdomains[dom['tag']] = []
        if dom['name'] not in tagdomains[dom['tag']]:
            tagdomains[dom['tag']].append(dom)


    testtagdomains = copy.deepcopy(atagdomains)


    print('testdomainnames: %d traindomainnames: %d' % (len(testdomainnames), len(traindomainnames)))
    print('num of train tables: %d test tables: %d' % (len(tbs), len(ttbs)))
    print('all domains: %d train domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d' % (len(alldomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))
    print('all domains: %d domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d'  % (len(alldomains), len(testdomains), len(keys), len(vecs), len(testtagdomains), len(testdomaintags)))

    # building test and train domain clouds
    alldomainclouds = orgk.make_cloud(simfile, 0.75)
    print('all domainclouds: %d' % len(alldomainclouds))

    for dom in domains:
        domainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in traindomainnames:
                domainclouds[dom['name']][cd] = cp
    print('done dom cloud of train')
    for dom in testdomains:
        testdomainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in testdomainnames:
                testdomainclouds[dom['name']][cd] = cp

    print('domainclouds: %d testdomainclouds: %d' % (len(domainclouds), len(testdomainclouds)))



def init_flat():
    print('init_flat')
    global domainclouds, keys, vecs, domains, tagdomains, domaintags
    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        if t.startswith('socrata_'):
            ks.append(t)
            vs.append(e)


    print('tag_embs: %d socrata tags: %d vecs: %d' % (len(tag_embs), len(vs), len(vs)))

    keys = ks
    vecs = np.array(vs)

    alldomains = json.load(open(DOMAIN_FILE))

    tbs = []
    domains = []
    seen = []
    domaintags = dict()
    tagdomains = dict()
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue

        if dom['name'] not in seen:
            domains.append(dom)
            seen.append(dom['name'])

        if dom['name'] not in domaintags:
            domaintags[dom['name']] = []
        if dom['tag'] not in domaintags[dom['name']]:
            domaintags[dom['name']].append(dom['tag'])

        table = dom['name'][:dom['name'].rfind('_')]
        if table not in tbs:
            tbs.append(table)


        if dom['tag'] not in tagdomains:
            tagdomains[dom['tag']] = []
        tagdomains[dom['tag']].append(dom)

    print('num of alli unique domains: %d tables: %d all domains: %d all tagdomains: %d' % (len(seen), len(tbs),len(domains), len(tagdomains)))

    domainclouds = orgk.make_cloud(simfile, 0.75)
    print('domainclouds: %d' % len(domainclouds))


def init():
    global keys, vecs, domains, tagdomains, domaintags
    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        if t.startswith('socrata_'):
            ks.append(t)
            vs.append(e)



    print('tag_embs: %d socrata tags: %d vecs: %d' % (len(tag_embs), len(vs), len(vs)))

    keys = ks
    vecs = np.array(vs)

    alldomains = json.load(open(DOMAIN_FILE))

    tbs = []
    seen = []
    adomaintags = dict()
    atagdomains = dict()
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue

        if dom['name'] not in seen:
            seen.append(dom['name'])

        if dom['name'] not in adomaintags:
            adomaintags[dom['name']] = []
        if dom['tag'] not in adomaintags[dom['name']]:
            adomaintags[dom['name']].append(dom['tag'])

        table = dom['name'][:dom['name'].rfind('_')]
        if table not in tbs:
            tbs.append(table)


        if dom['tag'] not in atagdomains:
            atagdomains[dom['tag']] = []
        atagdomains[dom['tag']].append(dom)

    print('num of alli unique domains: %d tables: %d all domains: %d all tagdomains: %d' % (len(seen), len(tbs),len(alldomains), len(atagdomains)))

    # random sample of tags
    tvs = list(atagdomains.keys())
    print('selected tags: %d' % len(tvs))
    tagdomains = copy.deepcopy(atagdomains)
    domains = alldomains
    print('selected tags: domains: %d vecs: %d keys: %d' % (len(domains), len(vecs), len(keys)))
    # sampling
    stagdomains, sdomains = orgs.stratified_sample(tagdomains, 0.5)
    print('sampled domains: %d sampled tagdomains: %d' % (len(sdomains), len(stagdomains)))
    tagdomains = copy.deepcopy(stagdomains)
    domaintags = copy.deepcopy(adomaintags)
    domains = []
    # making domains unique
    seen = []
    tbs = []
    sample_domain_names = []
    for domain in sdomains:
        sample_domain_names.append(domain['name'])
        if domain['name'] not in seen:
            domains.append(domain)
            seen.append(domain['name'])
            table = domain['name'][:domain['name'].rfind('_')]
            if table not in tbs:
                tbs.append(table)
    sample_domain_names = list(set(sample_domain_names))
    print('sample_domain_names: %d' % len(sample_domain_names))
    print('sampled tables: %d' % len(tbs))
    print('sampled domains: %d  unique domains: %d keys: %d vecs: %d sampled tagdomains: %d domaintags: %d' % (len(sdomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))
    json.dump(sample_domain_names, open(traindomainfile, 'w'))
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    json.dump(list(dims.values()), open(dimfile, 'w'))

    #orgk.all_pair_sim(alldomains, simfile)


def fix(g, hierarchy_name):
    print('fix')
    h, iteration_sps, iteration_ls, sps, dsps = orgf.fix_plus(g, domains, tagdomains, domainclouds, 'opendata', domaintags)

    print('fuzzy eval: ')
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_likelihood_fuzzy(h.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)

    json.dump(success_probs, open('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json', 'w'))
    json.dump(iteration_sps, open('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json', 'w'))
    json.dump(iteration_ls, open('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json', 'w'))

    fix_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json'
    fix_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'
    orgp.plot(fix_json, fix_pdf, 'Fixed Organization')

    sps_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json'
    sps_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'
    orgp.plot_fix(sps_json, sps_pdf, 'Average Success Prob', hierarchy_name)

    ls_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json'
    ls_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'
    orgp.plot_fix(ls_json, ls_pdf, 'Log Likelihood', hierarchy_name)

    print('printed to %s' % ('fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'))
    print('printed iteration success plot to %s' % ('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'))
    print('printed iteration likelihood plot to %s' % ('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'))

    return success_probs, h, domain_success_probs



def agg_fuzzy(suffix1, suffix2):
    print('agg_fuzzy')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    print('done clustering')
    print('initial hierarchy with %d nodes' % len(list(gp.nodes)))
    orgh.init(gp, dimdomains, tagdomains, simfile, tagdomsimfile)
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_likelihood_fuzzy(gp.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)

    return gp, success_probs, domain_success_probs


def multidimensional_hierarchy(dim_id):
    print('building one dim of a multidimensional hierarchy')
    success_probs_before = dict()
    success_probs_before_intersect = dict()
    success_probs_after = dict()
    success_probs_after_intersect = dict()


    ds = datetime.datetime.now()

    gp, sps, before_dsps = agg_fuzzy('agg'+str(dim_id)+'outof'+str(dim_num), '')
    for t, p in sps.items():
        if t not in success_probs_before:
            success_probs_before[t] = 0.0
            success_probs_before_intersect[t] = (1.0-p)
        success_probs_before[t] += p
        success_probs_before_intersect[t] *= (1.0-p)

    print('fixing')
    sps, fg, after_dsps = fix(gp, 'agg_'+str(dim_id)+'outof'+str(dim_num))
    sp = sum(list(sps.values()))/float(len(sps))
    print('sp of dim %d after fix is %f.' % (dim_id, sp))
    for t, p in sps.items():
        if t not in success_probs_after:
            success_probs_after[t] = 0.0
            success_probs_after_intersect[t] = (1.0-p)
        success_probs_after[t] += p
        success_probs_after_intersect[t] *= (1.0-p)

    de = datetime.datetime.now()
    delapsed = de - ds
    print('elapsed time of dim %d is %d' % (dim_id, int(delapsed.total_seconds() * 1000)))
    # evaluating test domains: domains that are not in the sample
    test_success, gp, test_success_probs, test_likelihood, test_domain_success_probs = orgh.get_success_prob_likelihood_fuzzy(fg, testdomains, testtagdomains, testdomainclouds, 'opendata', testdomaintags)
    # saving domain success probs of train and test
    json.dump(after_dsps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/train_domain_probs_' + str(dim_id) + '.json', 'w'))
    json.dump(test_domain_success_probs, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/test_domain_probs_' + str(dim_id) + '.json', 'w'))

    for t, p in success_probs_before.items():
        success_probs_before[t] = 1-success_probs_before_intersect[t]
        if success_probs_before[t] > 1.0:
            success_probs_before[t] = 1.0
    for t, p in success_probs_after.items():
        success_probs_after[t] =1.0-success_probs_after_intersect[t]
        if success_probs_after[t] > 1.0:
            success_probs_after[t] = 1.0


    before_sp = sum(list(success_probs_before.values()))/float(len(success_probs_before))
    print('success prob of multidimensions before fix: %f' % before_sp)


    after_sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of multidimensions after fix: %f' % after_sp)



def flat(suffix):
    print('flat')
    g = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs, keys)
    orgh.init(g, domains, tagdomains, simfile, tagdomsimfile)
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_likelihood_fuzzy(g.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)


    jsonfile = 'od_output/flat_dists_' + str(len(domains)) + '_' + suffix + '.json'
    json.dump(success_probs, open(jsonfile, 'w'))
    print('number of tables: %d' % len(success_probs))
    print('sp: %f' % (sum(list(success_probs.values()))/len(success_probs)))
    print('printed to %s' % jsonfile)



#orgt.tag_embs()


DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs'
TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_label_embs'
#TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/boosted_socrata_label_embs'
#DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/boosted_socrata_domain_embs'

#init_flat()
#flat('socrata')

#init()

init_load_dim(1)
print('-------------------')
multidimensional_hierarchy(1)


#multidimensional_hierarchy(6, 1)


#orgc.cmeans_clustering(keys, vecs)
#dims = orgh.get_dimensions(keys, vecs, 10)
#dims = orgh.get_dimensions_plus(keys, vecs)

