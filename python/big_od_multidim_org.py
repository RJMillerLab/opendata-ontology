from sklearn.cluster import KMeans
import random
import org.od_hierarchy as orgh
import org.graph as orgg
import org.cluster as orgc
import org.sample as orgs
import org.cloud as orgk
import org.od_fix as orgf
import numpy as np
import json
import copy
import datetime

simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json'
dimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_dims.json'
traindomainfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_sample_domain_names.json'
tagdomsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_domain_sims.json'
tagrepsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_rep_sims.json'
keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
domainclouds = dict()
domaintags = dict()
dimdomains = []
dimtagdomains = dict()
dimdomainclouds = dict()
dimdomaintags = dict()

reps = []
repdomains = dict()

dim_num = 10

def get_reps(rep_num):
    global repdomains, reps
    print('rep_num: %d' % rep_num)
    domvecs = [dom['mean'] for dom in domains]
    kmeans = KMeans(n_clusters=rep_num, random_state=random.randint(1,1000)).fit(domvecs)
    subs = dict()
    for i in range(len(kmeans.labels_)):
        c = kmeans.labels_[i]
        if c not in subs:
            subs[c] = []
        if 'centroid_' + str(c) not in repdomains:
            repdomains['centroid_' + str(c)] = []
        repdomains['centroid_' + str(c)].append(domains[i]['name'])
    print('kmeans.cluster_centers_: %d' % len(kmeans.cluster_centers_))
    for i in range(len(kmeans.cluster_centers_)):
        reps.append({'name': 'centroid_' + str(i), 'mean': list(kmeans.cluster_centers_[i])})
    print('repdomains: %d' % len(repdomains))
    print('tagdomains: %d' % len(tagdomains))



def init_dim(dim_id):

    global domainclouds, keys, vecs, domains, tagdomains,  domaintags, dimdomainclouds, dimdomaintags, dimdomains, dimtagdomains

    domaintags, tagdomains, domainclouds = dict(), dict(), dict()
    dimdomaintags, dimtagdomains, dimdomainclouds = dict(), dict(), dict()
    domains, dimdomains = [], []

    dims = json.load(open(dimfile, 'r'))
    dimtags = dims[dim_id]

    sample_domain_names = json.load(open(traindomainfile, 'r'))
    print('sample_domain_names: %d' % len(sample_domain_names))

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

    tbs, atbs = dict(), dict()
    trainseendoms, dimseendoms = dict(), dict()
    traindomainnames = dict()
    dimdomainnames = dict()
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue
        table = dom['name'][:dom['name'].rfind('_')]
        atbs[table] = True

        if dom['name'] not in dimseendoms:
            dimdomains.append(dom)
            dimseendoms[dom['name']] = True

        if dom['tag'] not in dimtagdomains:
            dimtagdomains[dom['tag']] = []
        if dom['name'] not in dimtagdomains[dom['tag']]:
            dimtagdomains[dom['tag']].append(dom['name'])

        if dom['name'] not in dimdomaintags:
            dimdomaintags[dom['name']] = []
        if dom['tag'] not in dimdomaintags[dom['name']]:
            dimdomaintags[dom['name']].append(dom['tag'])

        dimdomainnames[dom['name']] = True

        if dom['name'] not in sample_domain_names:
            continue

        traindomainnames[dom['name']] = True

        if dom['name'] not in trainseendoms:
            domains.append(dom)
            trainseendoms[dom['name']] = True

        if dom['name'] not in domaintags:
            domaintags[dom['name']] = []
        if dom['tag'] not in domaintags[dom['name']]:
            domaintags[dom['name']].append(dom['tag'])

        tbs[table] = True


        if dom['tag'] not in tagdomains:
            tagdomains[dom['tag']] = []
        if dom['name'] not in tagdomains[dom['tag']]:
            tagdomains[dom['tag']].append(dom['name'])



    print('dimdomains: %d traindomainnames: %d' % (len(domains), len(traindomainnames)))
    print('num of train tables: %d number of dim tables: %d' % (len(tbs), len(atbs)))
    print('all domains: %d train domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d' % (len(alldomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))
    print('all domains: %d dim domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d'  % (len(alldomains), len(dimdomains), len(keys), len(vecs), len(dimtagdomains), len(dimdomaintags)))

    # building test and train domain clouds
    alldomainclouds = orgk.make_cloud(simfile, 0.75)
    print('all domainclouds: %d' % len(alldomainclouds))

    for dom in domains:
        domainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in traindomainnames:
                domainclouds[dom['name']][cd] = cp
    print('done dom cloud of train')
    for dom in dimdomains:
        dimdomainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in dimdomainnames:
                dimdomainclouds[dom['name']][cd] = cp

    print('domainclouds: %d dimdomainclouds: %d' % (len(domainclouds), len(dimdomainclouds)))

    num_rep = max(50, int(len(domains)/10.0))
    get_reps(num_rep)

    #print('calculating tag rep sims')
    #orgh.get_tag_domain_sim(reps, keys, vecs, tagrepsimfile)


    # saving all indices and maps to file
    #domainclouds, keys, vecs, domains, tagdomains,  domaintags, dimdomainclouds, dimdomaintags, dimdomains,         dimtagdomains
    print('saving dim data to files.')
    json.dump(domainclouds, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomainclouds_'+str(dim_id)+'.json', 'w'))
    json.dump(domains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomains_'+str(dim_id)+'.json', 'w'))
    json.dump(tagdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traintagdomains_'+str(dim_id)+'.json', 'w'))
    json.dump(domaintags, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomaintags_'+str(dim_id)+'.json', 'w'))

    json.dump(dimdomainclouds, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomainclouds_'+str(dim_id)+'.json', 'w'))
    json.dump(dimdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomains_'+str(dim_id)+'.json', 'w'))
    json.dump(dimtagdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimtagdomains_'+str(dim_id)+'.json', 'w'))
    json.dump(dimdomaintags, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomaintags_'+str(dim_id)+'.json', 'w'))

    json.dump(keys, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagnames_'+str(dim_id)+'.json', 'w'))
    json.dump([list(v) for v in vecs], open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagvecs_'+str(dim_id)+'.json', 'w'))

    json.dump(reps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/reps_'+str(dim_id)+'.json', 'w'))
    json.dump(repdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/repdomains_'+str(dim_id)+'.json', 'w'))
    print('done saving.')





def load_dim(dim_id):
    global domainclouds, keys, vecs, domains, tagdomains,  domaintags, dimdomainclouds, dimdomaintags, dimdomains,  dimtagdomains, reps, repdomains

    start = datetime.datetime.now()
    domainclouds = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomainclouds_'+str(dim_id)+'.json', 'r'))
    domains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomains_'+str(dim_id)+'.json', 'r'))
    tagdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traintagdomains_'+str(dim_id)+'.json', 'r'))
    domaintags = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomaintags_'+str(dim_id)+'.json', 'r'))

    dimdomainclouds = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomainclouds_'+str(dim_id)+'.json', 'r'))
    dimdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomains_'+str(dim_id)+'.json', 'r'))
    dimtagdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimtagdomains_'+str(dim_id)+'.json', 'r'))
    dimdomaintags = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomaintags_'+str(dim_id)+'.json', 'r'))

    keys = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagnames_'+str(dim_id)+'.json', 'r'))
    vecs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagvecs_'+str(dim_id)+'.json', 'r'))
    vecs = np.array([np.array(v) for v in vecs])

    reps = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/reps_'+str(dim_id)+'.json', 'r'))
    repdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/repdomains_'+str(dim_id)+'.json', 'r'))

    print('loaded all dim files in %d ms.' % ((datetime.datetime.now()-start).total_seconds()*1000))

    print('reps: %d repdomains: %d tagdomains: %d domaintags: %d domainclouds: %d' % (len(reps), len(repdomains), len(tagdomains), len(domaintags), len(domainclouds)))
    print('keys: %d vecs: %d' % (len(keys), len(vecs)))
    print('dim domains: %d domaintags: %d tagdomains: %d domainclouds: %d' % (len(dimdomains), len(dimdomaintags), len(dimtagdomains), len(dimdomainclouds)))



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
        tagdomains[dom['tag']].append(dom['name'])

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
    atagdomains = dict()
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue

        if dom['name'] not in seen:
            seen.append(dom['name'])

        if dom['tag'] not in atagdomains:
            atagdomains[dom['tag']] = []
        atagdomains[dom['tag']].append(dom['name'])

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
    sample_domain_names = list(set(sample_domain_names))
    print('sample_domain_names: %d' % len(sample_domain_names))
    print('sampled tables: %d' % len(tbs))
    print('sampled domains: %d  unique domains: %d keys: %d vecs: %d sampled tagdomains: %d domaintags: %d' % (len(sdomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))
    json.dump(sample_domain_names, open(traindomainfile, 'w'))
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    json.dump(list(dims.values()), open(dimfile, 'w'))

    #orgk.all_pair_sim(alldomains, simfile)
    # Tag-rep and Tag-domain sims are precalculated.
    #orgh.get_tag_domain_sim(domains, keys, vecs, tagdomsimfile)
    #orgh.get_tag_domain_sim(reps, keys, vecs, tagrepsimfile)


def fix(g, hierarchy_name):
    print('fix')
    h, iteration_sps, iteration_ls, sps, dsps = orgf.fix_plus(g, domains, tagdomains, domainclouds, 'opendata', domaintags, reps, repdomains)

    print('fuzzy eval: ')
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(h.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)

    json.dump(domain_success_probs, open('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_domain_sps.json', 'w'))
    json.dump(success_probs, open('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_table_sps.json', 'w'))

    print('printed to %s' % ('fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'))
    print('saved domain success probs to %s' % ('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_domain_sps.json'))
    print('saved table succsss probs to %s' % ('od_output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_table_sps.json'))

    return success_probs, h, domain_success_probs



def agg_fuzzy(suffix1, dim_id):
    print('agg_fuzzy')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    print('done clustering')
    print('initial hierarchy with %d nodes' % len(list(gp.nodes)))
    orgh.get_state_domain_sims(gp, tagrepsimfile, reps)

    domainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomains_'+str(dim_id)+'.json'

    orgh.init(gp, domainsfile, simfile)

    tagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traintagdomains_'+str(dim_id)+'.json'
    domaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomainclouds_'+str(dim_id)+'.json'
    domaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomaintags_'+str(dim_id)+'.json'
    repdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/repdomains_'+str(dim_id)+'.json'
    repsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/reps_'+str(dim_id)+'.json'

    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_rep_prob_fuzzy(gp.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags, repdomains, reps)
    json.dump(domain_success_probs, open('od_output/agg_' + suffix1 + '_' + str(len(domains)) + '_domain_sps.json', 'w'))
    json.dump(success_probs, open('od_output/agg_' + suffix1 + '_' + str(len(domains)) + '_table_sps.json', 'w'))
    print('initial dsps to %s' % ('od_output/agg_' + suffix1 + '_' + str(len(domains)) + '_domain_sps.json'))

    return gp, success_probs, domain_success_probs


def multidimensional_hierarchy(dim_id):
    print('building one dim of a multidimensional hierarchy')
    success_probs_before = dict()
    success_probs_before_intersect = dict()
    success_probs_after = dict()
    success_probs_after_intersect = dict()


    ds = datetime.datetime.now()

    gp, sps, before_dsps = agg_fuzzy('agg'+str(dim_id)+'of'+str(dim_num), dim_id)
    for t, p in sps.items():
        if t not in success_probs_before:
            success_probs_before[t] = 0.0
            success_probs_before_intersect[t] = (1.0-p)
        success_probs_before[t] += p
        success_probs_before_intersect[t] *= (1.0-p)

    print('fixing')
    sps, fg, after_dsps = fix(gp, 'agg_'+str(dim_id)+'of'+str(dim_num))
    sp = sum(list(sps.values()))/float(len(sps))
    print('sp of dim %d for domains after fix is %f.' % (dim_id, sp))
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
    # need to call init() to reindex all domains on name
    orgh.get_state_domain_sims(fg, tagdomsimfile, dimdomains)
    dimdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomains_'+str(dim_id)+'.json'
    orgh.init(fg, dimdomainsfile, simfile)
    print('dimdomains in main: %d vs %d dimtagdomains: %d vs %d' % (len(dimdomains), len(domains), len(dimtagdomains), len(tagdomains)))
    test_success, gp, test_success_probs, test_likelihood, test_domain_success_probs = orgh.get_success_prob_fuzzy(fg, dimdomains, dimtagdomains, dimdomainclouds, 'opendata', dimdomaintags)
    print('sp of dim %d for all domains: %f' % (dim_id, test_success))
    # saving domain success probs of train and test
    json.dump(after_dsps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/train_domain_probs_' + str(dim_id) + '.json', 'w'))
    json.dump(test_domain_success_probs, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/all_domain_probs_' + str(dim_id) + '.json', 'w'))
    json.dump(test_domain_success_probs, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/all_table_probs_' + str(dim_id) + '.json', 'w'))

    for t, p in success_probs_before.items():
        success_probs_before[t] = 1-success_probs_before_intersect[t]
        if success_probs_before[t] > 1.0:
            success_probs_before[t] = 1.0
    for t, p in success_probs_after.items():
        success_probs_after[t] =1.0-success_probs_after_intersect[t]
        if success_probs_after[t] > 1.0:
            success_probs_after[t] = 1.0


    before_sp = sum(list(success_probs_before.values()))/float(len(success_probs_before))
    print('success prob of dim %d before fix for train domains: %f' % (dim_id, before_sp))


    after_sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of dim %d after fix with %d reps: %f' % (dim_id, len(reps), after_sp))

    print('success prob of dim %d for all domains after fix: %f' % (dim_id, test_success))



def flat(suffix):
    print('flat')
    g = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs, keys)
    orgh.get_state_domain_sims(g, tagrepsimfile, domains)
    orgh.init(g, domains, tagdomains, simfile)
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(g.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)


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

#init_dim(8)
load_dim(8)
print('-------------------')
multidimensional_hierarchy(8)


