from sklearn.cluster import KMeans
import org.semantic as orgm
import random
import org.od_hierarchy as orgh
import org.visualize as orgv
import org.graph as orgg
import org.cluster as orgc
import org.cloud as orgk
import org.od_fix as orgf
import numpy as np
import json
import datetime

repsfile, tagtablesfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile, dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile = '', '', '', '', '', '', '', '', '', '', ''


simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json'
dimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/od_dims.json'
studydomainfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/study_domain_names.json'
tagdomsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/tag_domain_sims.json'
tagrepsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/tag_rep_sims.json'
tagdomtransprobsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/tag_dom_trans_probs.json'
keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
tagtables = dict()
domainclouds = dict()
domaintags = dict()
dimdomains = []
dimtagdomains = dict()
dimdomainclouds = dict()
dimdomaintags = dict()
reps = []
repdomains = dict()

dim_num = 3

def get_reps(rep_num, dim_id, domains):
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
    json.dump(reps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/reps_kmeans_'+str(dim_id)+'.json', 'w'))
    json.dump(repdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/repdomains_kmeans_'+str(dim_id)+'.json', 'w'))



def init_dim(dim_id):

    print('init_dim %d' % dim_id)

    global keys, vecs, dimtagtables, dimdomainclouds, dimdomaintags, dimdomains, dimtagdomains

    dimtagtables, dimdomaintags, dimtagdomains, dimdomainclouds = dict(), dict(), dict(), dict()
    dimdomains = []

    dims = json.load(open(dimfile, 'r'))
    dimtags = dims[dim_id]

    study_domain_names = json.load(open(studydomainfile, 'r'))
    print('study_domain_names: %d' % len(study_domain_names))

    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        ct = orgh.clean_tag(t)
        if ct in dimtags and ct not in ks:
            ks.append(ct)
            vs.append(e)

    print('tag_embs: %d dim ks: %d vs: %d' % (len(tag_embs), len(vs), len(vs)))

    keys = ks
    vecs = np.array(vs)

    alldomains = json.load(open(DOMAIN_FILE))

    atbs =  dict()
    dimseendoms = dict()
    dimdomainnames = dict()
    for dom in alldomains:
        domtag = orgh.clean_tag(dom['tag'])
        if domtag not in ks:
            continue
        if dom['name'] not in study_domain_names:
            continue
        table = dom['name'][:dom['name'].rfind('_')]
        atbs[table] = True

        if dom['name'] not in dimseendoms:
            dimdomains.append(dom)
            dimseendoms[dom['name']] = True

        if domtag not in dimtagdomains:
            dimtagdomains[domtag] = []
            dimtagtables[domtag] = []
        if dom['name'] not in dimtagdomains[domtag]:
            dimtagdomains[domtag].append(dom['name'])
        if table not in dimtagtables[domtag]:
            dimtagtables[domtag].append(table)

        if dom['name'] not in dimdomaintags:
            dimdomaintags[dom['name']] = []
        if domtag not in dimdomaintags[dom['name']]:
            dimdomaintags[dom['name']].append(domtag)

        dimdomainnames[dom['name']] = True

    print('dimtagtables: %d' % len(dimtagtables))
    print('dimdomains: %d, dimdomaintags: %d, dimtables: %d' % (len(dimdomains), len(dimdomaintags), len(atbs)))

    print('all domains: %d dim domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d'  % (len(alldomains), len(dimdomains), len(keys), len(vecs), len(dimtagdomains), len(dimdomaintags)))

    alldomainclouds = orgk.make_cloud(simfile, 0.80)
    print('all domainclouds: %d' % len(alldomainclouds))

    for dom in dimdomains:
        dimdomainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in dimdomainnames:
                dimdomainclouds[dom['name']][cd] = cp

    print('dimdomainclouds: %d' % (len(dimdomainclouds)))

    num_rep = min(len(ks), int(len(dimdomains)/10.0))
    get_reps(num_rep, dim_id, dimdomains)

    # saving all indices and maps to file
    print('saving dim data to files.')
    json.dump(dimdomainclouds, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomainclouds_'+str(dim_id)+'.json', 'w'))
    json.dump(dimdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomains_'+str(dim_id)+'.json', 'w'))
    json.dump(dimtagtables, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimtagtables_'+  str(dim_id)+'.json', 'w'))
    json.dump(dimtagdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimtagdomains_'+str(dim_id)+'.json', 'w'))
    json.dump(dimdomaintags, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomaintags_'+str(dim_id)+'.json', 'w'))
    json.dump(keys, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagnames_'+str(dim_id)+'.json', 'w'))
    json.dump([list(v) for v in vecs], open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagvecs_'+str(dim_id)+'.json', 'w'))

    json.dump(reps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/reps_'+str(dim_id)+'.json', 'w'))
    json.dump(repdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/repdomains_'+str(dim_id)+'.json', 'w'))
    print('done saving.')

    # Tag-rep and Tag-domain sims are precalculated.
    print('get_tag_domain_sim')
    dim_tagdomsimfile = tagdomsimfile.replace('.json', '_'+str(dim_id)+'.json')
    orgh.get_tag_domain_sim(dimdomains, keys, vecs, dim_tagdomsimfile)

    print('get_tag_domain_transprobs for dim')
    dim_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_dim'+ str(dim_id)+'.json')
    orgh.get_tag_domain_trans_probs(dim_tagdomsimfile, dimtagdomains,dim_tagdomtransprobsfile)
    print('done')



def load_dim(dim_id):
    global domainsfile, repsfile, tagtablesfile, repdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile, domainclouds, keys, vecs, domains, tagtables, tagdomains,  domaintags, reps, repdomains

    start = datetime.datetime.now()

    domaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomainclouds_'+str(dim_id)+'.json'
    domainclouds = json.load(open(domaincloudsfile))
    domainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomains_'+str(dim_id)+'.json'
    domains = json.load(open(domainsfile))
    tagtablesfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimtagtables_'+ str(dim_id)+'.json'
    tagtables = json.load(open(tagtablesfile))
    tagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimtagdomains_'+            str(dim_id)+'.json'
    tagdomains = json.load(open(tagdomainsfile))
    domaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomaintags_'+            str(dim_id)+'.json'
    domaintags = json.load(open(domaintagsfile))

    keys = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagnames_'+str(dim_id)+'.json', 'r'))
    vecs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagvecs_'+str(dim_id)+'.json', 'r'))
    vecs = np.array([np.array(v) for v in vecs])

    repsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/reps_'+str(dim_id)+'.json'
    reps = json.load(open(repsfile))
    repdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/repdomains_'+str(dim_id)+'.json'
    repdomains = json.load(open(repdomainsfile))

    print('loaded all dim files in %d ms.' % ((datetime.datetime.now()-start).total_seconds()*1000))

    print('reps: %d repdomains: %d tagdomains: %d domaintags: %d domainclouds: %d' % (len(reps), len(repdomains), len(tagdomains), len(domaintags), len(domainclouds)))
    print('keys: %d vecs: %d' % (len(keys), len(vecs)))


def init():
    print('init()')
    global keys, vecs, domains, tagdomains
    tag_embs = json.load(open(TAG_EMB_FILE))
    study_tags = json.load(open(STUDY_TAG_FILE))
    alldomains = json.load(open(DOMAIN_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        ct = orgh.clean_tag(t)
        if ct in study_tags and ct not in ks:
            ks.append(ct)
            vs.append(e)
    print('tag_embs: %d clean tags: %d vecs: %d' % (len(tag_embs), len(vs), len(vs)))
    keys = ks
    vecs = np.array(vs)
    tbs = dict()
    seen = []
    for dom in alldomains:
        domtag = orgh.clean_tag(dom['tag'])
        if domtag not in ks:
            continue
        if dom['name'] not in seen:
            seen.append(dom['name'])
        domains.append(dom)
        table = dom['name'][:dom['name'].rfind('_')]
        tbs[table]= True
        if domtag not in tagdomains:
            tagdomains[domtag] = []
        tagdomains[domtag].append(dom)

    print('num of unique domains: %d tables: %d domains: %d all tagdomains: %d' % (len(seen), len(tbs),len(domains), len(tagdomains)))

    print('writing domain names')
    json.dump(seen, open(studydomainfile, 'w'))
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    json.dump(list(dims.values()), open(dimfile, 'w'))
    print('created %d dims' % len(dims))

    print('done writing dom files')


def fix(g, hierarchy_name, dim_id):
    print('fix')

    global domainsfile, repsfile, repdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    h, iteration_sps, iteration_ls, sps, dsps, iteration_ts = orgf.fix_plus(g, domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile, repsfile, repdomainsfile)

    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(h.copy(), domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile)

    print('success prob for train domains after fix: %f(%f)' % (max_success, sum(list(success_probs.values()))/len(success_probs)))

    return success_probs, h, domain_success_probs, iteration_ls, iteration_ts



def agg_fuzzy(suffix, dim_id):
    print('agg_fuzzy')
    global domainsfile, repsfile, repdomainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    print('initial hierarchy with %d nodes' % len(list(gp.nodes)))
    #dim_tagdomsimfile = tagdomsimfile.replace('.json', '_'+str(dim_id)+'.json')
    #orgh.get_state_domain_sims(gp, dim_tagdomsimfile, domainsfile)
    #train_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_dim' + str(dim_id)+'.json')

    #orgh.init(gp, domainsfile, simfile, train_tagdomtransprobsfile)

    #max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(gp.copy(), domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile)

    #print('the success prob of the agg org: %f' % max_success)

    #json.dump(domain_success_probs, open('study_output/agg_' + suffix + '_' + str(len(domains)) + '_domain_sps.json', 'w'))
    #json.dump(success_probs, open('study_output/agg_' + suffix + '_' + str(len(domains)) + '_table_sps.json', 'w'))
    #print('initial dsps to %s' % ('study_output/agg_' + suffix + '_' + str(len(domains)) + '_domain_sps.json'))

    #return gp, success_probs, domain_success_probs
    return gp, [], []


def multidimensional_hierarchy(dim_id):
    print('building one dim of a multidimensional hierarchy')

    ds = datetime.datetime.now()
    print('start time:')
    print(ds)

    # initial org
    gp, before_sps, before_dsps = agg_fuzzy(str(dim_id)+'of'+str(dim_num), dim_id)
    print('sp of initial org on train: %f' % (sum(list(before_sps.values()))/len(before_sps)))

    # improving the org
    print('fixing')
    #after_sps, fg, after_dsps, ls, ts = fix(gp, 'fix_'+str(dim_id)+'of'+str(dim_num), dim_id)

    #print('time to build dim %d is %d' % (dim_id, int((datetime.datetime.now()-ds).total_seconds() * 1000)))


    print('extracting semantics of org')
    orgm.init_fromarrays(keys, vecs, tagtables)
    semfilename = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org'+str(dim_id)+'of'+str(dim_num)+'.sem'
    #sg = orgm.org_with_semantic(fg, semfilename)
    sg = orgm.org_with_semantic(gp, semfilename)
    visfilename = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org'+str(dim_id)+'of'+str(dim_num)+'_vis.json'
    nodesfilename = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org'+str(dim_id)+'of'+str(dim_num)+'_vis.csv'
    orgv.save_to_visualize(sg, visfilename, nodesfilename, tagtablesfile)

    # need to call init() to reindex all domains on name
    #print('success prob of dim %d before fix for train domains: %f' % (dim_id, (sum(list(before_sps.values()))/len(before_sps))))

    #print('success prob of dim %d after fix with %d reps: %f' % (dim_id, len(reps), (sum(list(after_sps.values()))/len(after_sps))))

    print('end time:')
    print(datetime.datetime.now())



def get_org_success_probs(dimsfilename, spsfilename):
    with open(dimsfilename) as df:
        spsfiles = df.read().splitlines()
    org_sps = dict()
    for spsf in spsfiles:
        dimsps = json.load(open(spsf, 'r'))
        for t, p in dimsps.items():
            if t not in org_sps:
                org_sps[t] = (1.0-p)
            else:
                org_sps[t] *= (1.0-p)
    for t, p in org_sps.items():
        org_sps[t] = 1.0-p
    json.dump(org_sps, open(spsfilename, 'w'))


DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs'
TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_label_embs'
STUDY_TAG_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/socrata/clean_tags.json'
#STUDY_TAG_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/socrata/aug_clean_tags.json'

#init()
#init_dim(0)
load_dim(0)
multidimensional_hierarchy(0)
print('-------------------')
