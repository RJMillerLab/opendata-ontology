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

domainsfile = ''
repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile, dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile = '', '', '', '', '', '', '', '', '', ''

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
domainclouds = dict()
domaintags = dict()
dimdomains = []
dimtagdomains = dict()
dimdomainclouds = dict()
dimdomaintags = dict()

reps = []
repdomains = dict()

dim_num = 10

def get_reps(rep_num, dim_id):
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

    global domainclouds, keys, vecs, domains, tagdomains,  domaintags, dimdomainclouds, dimdomaintags, dimdomains, dimtagdomains

    domaintags, tagdomains, domainclouds = dict(), dict(), dict()
    dimdomaintags, dimtagdomains, dimdomainclouds = dict(), dict(), dict()
    domains, dimdomains = [], []

    dims = json.load(open(dimfile, 'r'))
    dimtags = dims[dim_id]

    sample_domain_names = json.load(open(studydomainfile, 'r'))
    print('sample_domain_names: %d' % len(sample_domain_names))

    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        if t in dimtags:
            ks.append(t)
            vs.append(e)

    print('tag_embs: %d dim ks: %d vs: %d' % (len(tag_embs), len(vs), len(vs)))

    keys = ks
    vecs = np.array(vs)

    alldomains = json.load(open(DOMAIN_FILE))

    tbs, atbs = dict(), dict()
    trainseendoms, dimseendoms = dict(), dict()
    studydomainnames, dimdomainnames = dict(), dict()
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

        studydomainnames[dom['name']] = True

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

    print('dimdomains: %d, dimdomaintags: %d, dimtables: %d' % (len(dimdomains), len(dimdomaintags), len(atbs)))
    return

    print('dimdomains: %d studydomainnames: %d' % (len(domains), len(studydomainnames)))
    print('num of train tables: %d number of dim tables: %d' % (len(tbs), len(atbs)))
    print('all domains: %d train domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d' % (len(alldomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))
    print('all domains: %d dim domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d'  % (len(alldomains), len(dimdomains), len(keys), len(vecs), len(dimtagdomains), len(dimdomaintags)))

    alldomainclouds = orgk.make_cloud(simfile, 0.75)
    print('all domainclouds: %d' % len(alldomainclouds))

    for dom in domains:
        domainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in studydomainnames:
                domainclouds[dom['name']][cd] = cp
    print('done dim cloud')
    for dom in dimdomains:
        dimdomainclouds[dom['name']] = dict()
        for cd, cp in alldomainclouds[dom['name']].items():
            if cd in dimdomainnames:
                dimdomainclouds[dom['name']][cd] = cp

    print('domainclouds: %d dimdomainclouds: %d' % (len(domainclouds), len(dimdomainclouds)))

    num_rep = min(len(ks), int(len(domains)/10.0))
    get_reps(num_rep, dim_id)

    # saving all indices and maps to file
    print('saving dim data to files.')
    json.dump(domainclouds, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomainclouds_'+str(dim_id)+'.json', 'w'))
    json.dump(domains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomains_'+str(dim_id)+'.json', 'w'))
    json.dump(tagdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traintagdomains_'+str(dim_id)+'.json', 'w'))
    json.dump(domaintags, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomaintags_'+str(dim_id)+'.json', 'w'))

    json.dump(dimdomainclouds, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomainclouds_'+str(dim_id)+'.json', 'w'))
    json.dump(dimdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomains_'+str(dim_id)+'.json', 'w'))
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

    print('get_tag_domain_transprobs for train')
    train_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_tr'+str(dim_id)+'.json')
    orgh.get_tag_domain_trans_probs(dim_tagdomsimfile, tagdomains, train_tagdomtransprobsfile)
    print('get_tag_domain_transprobs for test')
    all_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_all'+ str(dim_id)+'.json')
    orgh.get_tag_domain_trans_probs(dim_tagdomsimfile, dimtagdomains,all_tagdomtransprobsfile)
    print('done')



def load_dim(dim_id):
    global domainclouds, keys, vecs, domains, tagdomains,  domaintags, dimdomainclouds, dimdomaintags, dimdomains,  dimtagdomains, reps, repdomains

    start = datetime.datetime.now()
    domainclouds = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomainclouds_'+str(dim_id)+'.json', 'r'))
    domains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomains_'+str(dim_id)+'.json', 'r'))
    tagdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traintagdomains_'+str(dim_id)+'.json', 'r'))
    domaintags = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomaintags_'+str(dim_id)+'.json', 'r'))

    dimdomainclouds = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomainclouds_'+str(dim_id)+'.json', 'r'))
    dimdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomains_'+str(dim_id)+'.json', 'r'))
    dimtagdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimtagdomains_'+str(dim_id)+'.json', 'r'))
    dimdomaintags = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomaintags_'+str(dim_id)+'.json', 'r'))

    keys = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagnames_'+str(dim_id)+'.json', 'r'))
    vecs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagvecs_'+str(dim_id)+'.json', 'r'))
    vecs = np.array([np.array(v) for v in vecs])

    reps = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/reps_'+str(dim_id)+'.json', 'r'))
    repdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/repdomains_'+str(dim_id)+'.json', 'r'))

    print('loaded all dim files in %d ms.' % ((datetime.datetime.now()-start).total_seconds()*1000))

    print('reps: %d repdomains: %d tagdomains: %d domaintags: %d domainclouds: %d' % (len(reps), len(repdomains), len(tagdomains), len(domaintags), len(domainclouds)))
    print('keys: %d vecs: %d' % (len(keys), len(vecs)))
    print('dim domains: %d domaintags: %d tagdomains: %d domainclouds: %d' % (len(dimdomains), len(dimdomaintags), len(dimtagdomains), len(dimdomainclouds)))




def read_dim(dim_id):
    print('dim_id: %d' % dim_id)
    global keys, vecs, reps, repdomains, domainsfile, repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile, dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    print('assigning file names')

    domainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomains_'+str(dim_id)+'.json'
    tagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traintagdomains_'+str(dim_id)+'.json'
    domaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomainclouds_'+str(dim_id)+'.json'
    domaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/traindomaintags_'+str(dim_id)+'.json'
    dimdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomains_'+str(dim_id)+'.json'
    dimtagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimtagdomains_'+str(dim_id)+'.json'
    dimdomaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomainclouds_'+str(dim_id)+'.json'
    dimdomaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/dimdomaintags_'+str(dim_id)+'.json'
    repsfile =  repdomainsfile

    repsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/reps_'+          str(dim_id)+'.json'
    repdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/repdomains_'+str(dim_id)+'.json'

    keys = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagnames_'+str(dim_id)+'.json', 'r'))
    vecs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/tagvecs_'+str(dim_id)+'.json', 'r'))
    vecs = np.array([np.array(v) for v in vecs])

    reps = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/reps_'+str(dim_id)+'.json', 'r'))
    repdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_tmp/repdomains_'+str(dim_id)+'.json', 'r'))

    print('number of reps: %d' % len(reps))
    print('number of keys: %d' % len(keys))
    print('number of vecs: %d' % len(vecs))

    ds = json.load(open(dimdomainsfile))
    tables = dict()
    label_tables = dict()
    table_labels = dict()
    for dom in ds:
        t = dom['name'][:dom['name'].rfind('_')]
        tables[t] = True
        tag = dom['tag']
        if tag not in label_tables:
            label_tables[tag] = [t]
        else:
            label_tables[tag].append(t)
        if t not in table_labels:
            table_labels[t] = [dom['tag']]
        else:
            table_labels[t].append(dom['tag'])
    json.dump(list(tables.keys()), open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/tables_'+str(len(tables))+'.json', 'w'))
    json.dump(table_labels, open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/table_labels_'+str(len(tables))+'.json', 'w'))
    json.dump(label_tables, open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/ekg/label_tables_'+str(len(tables))+'.json', 'w'))
    print('number of tables: %d' % len(tables))


def init():
    global keys, vecs, domains, tagdomains
    tag_embs = json.load(open(TAG_EMB_FILE))
    study_tags = json.load(open(STUDY_TAG_FILE))
    alldomains = json.load(open(DOMAIN_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        if t in study_tags:
            ks.append(t)
            vs.append(e)
    print('tag_embs: %d clean tags: %d vecs: %d' % (len(tag_embs), len(vs), len(vs)))
    keys = ks
    vecs = np.array(vs)
    tbs = dict()
    seen = []
    for dom in alldomains:
        if dom['tag'] not in ks:
            continue
        if dom['name'] not in seen:
            seen.append(dom['name'])
        domains.append(dom)
        table = dom['name'][:dom['name'].rfind('_')]
        tbs[table]= True
        if dom['tag'] not in tagdomains:
            tagdomains[dom['tag']] = []
        tagdomains[dom['tag']].append(dom)

    print('num of unique domains: %d tables: %d domains: %d all tagdomains: %d' % (len(seen), len(tbs),len(domains), len(tagdomains)))

    print('writing domain names')
    json.dump(list(seen.keys()), open(studydomainfile, 'w'))
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    json.dump(list(dims.values()), open(dimfile, 'w'))
    print('created %d dims' % len(dims))

    #json.dump(domains, open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_'+str(len(domains))+'_embs.json', 'w'))

    print('done writing dom files')

    #orgk.all_pair_sim(alldomains, simfile)
    # Tag-rep and Tag-domain sims are precalculated.
    #print('get_tag_domain_sim')
    #orgh.get_tag_domain_sim(domains, keys, vecs, tagdomsimfile)
    #print('get_tag_domain_transprobs')
    #orgh.get_tag_domain_trans_probs(tagdomsimfile, atagdomains, tagdomtransprobsfile)


def fix(g, hierarchy_name, dim_id):
    print('fix')

    global domainsfile, repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile, dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    h, iteration_sps, iteration_ls, sps, dsps, iteration_ts = orgf.fix_plus(g, domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile, repsfile, repdomainsfile)

    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(h.copy(), domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile)

    print('success prob for train domains after fix: %f(%f)' % (max_success, sum(list(success_probs.values()))/len(success_probs)))

    return success_probs, h, domain_success_probs, iteration_ls, iteration_ts



def agg_fuzzy(suffix1, dim_id):
    print('agg_fuzzy')
    global domainsfile, repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile,dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    print('initial hierarchy with %d nodes' % len(list(gp.nodes)))
    dim_tagdomsimfile = tagdomsimfile.replace('.json', '_'+str(dim_id)+'.json')
    orgh.get_state_domain_sims(gp, dim_tagdomsimfile, domainsfile)
    train_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_tr'+       str(dim_id)+'.json')

    orgh.init(gp, domainsfile, simfile, train_tagdomtransprobsfile)

    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(gp.copy(), domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile)

    print('the success prob of the agg org: %f' % max_success)

    json.dump(domain_success_probs, open('study_output/agg_' + suffix1 + '_' + str(len(domains)) + '_domain_sps.json', 'w'))
    json.dump(success_probs, open('study_output/agg_' + suffix1 + '_' + str(len(domains)) + '_table_sps.json', 'w'))
    print('initial dsps to %s' % ('study_output/agg_' + suffix1 + '_' + str(len(domains)) + '_domain_sps.json'))

    return gp, success_probs, domain_success_probs


def multidimensional_hierarchy(dim_id):
    print('building one dim of a multidimensional hierarchy')

    ds = datetime.datetime.now()
    print('start time:')
    print(ds)

    # initial org
    gp, before_sps, before_dsps = agg_fuzzy('agg'+str(dim_id)+'of'+str(dim_num), dim_id)
    print('sp of initial org on train: %f' % (sum(list(before_sps.values()))/len(before_sps)))

    # improving the org
    print('fixing')
    after_sps, fg, after_dsps, ls, ts = fix(gp, 'agg_'+str(dim_id)+'of'+str(dim_num), dim_id)

    print('time to build dim %d is %d' % (dim_id, int((datetime.datetime.now()-ds).total_seconds() * 1000)))


    # saving the org
    orgfilname = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org'+str(dim_id)+'of'+str(dim_num)+'.txt'
    print('saving org to %s' % orgfilname)
    orgh.save(fg, orgfilname)

    orgm.init_fromarrays(keys, vecs)
    print('extracting semantics of org')
    semfilename = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org'+str(dim_id)+'of'+str(dim_num)+'.sem'
    #orgm.get_org_semantic_btree(orgfilname, semfilename)
    sg = orgm.org_with_semantic_btree(orgfilname, semfilename)
    # save org for visualization
    orgv.save_to_visualize(sg)
    # evaluating test domains: domains that are not in the sample
    # need to call init() to reindex all domains on name
    orgh.extend_node_dom_sims(fg, dimdomainsfile)

    all_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_all'+             str(dim_id)+'.json')
    orgh.init(fg, dimdomainsfile, simfile, all_tagdomtransprobsfile)


    test_success, gp, test_sps, test_likelihood, test_dsps = orgh.get_success_prob_fuzzy(fg, dimdomainsfile, dimtagdomainsfile, dimdomaincloudsfile, 'opendata', dimdomaintagsfile)
    print('success prob of dim %d for all domains: %f' % (dim_id, test_success))

    print('success prob of dim %d before fix for train domains: %f' % (dim_id, (sum(list(before_sps.values()))/len(before_sps))))

    print('success prob of dim %d after fix with %d reps: %f' % (dim_id, len(reps), (sum(list(after_sps.values()))/len(after_sps))))

    print('success prob of dim %d for all domains after fix: %f' % (dim_id, (sum(list(test_sps.values()))/len(test_sps))))

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

#init()

#init_dim(0)
#load_dim(6)
read_dim(5)
orgm.init_fromarrays(keys, vecs)
#orgm.get_org_semantic_btree('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org5of10.txt', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org5of10.sem')
sg = orgm.org_with_semantic_btree('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org5of10.txt', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org5of10.sem')
visfilename = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org5of10_vis.json'
nodesfilename = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/study_output/org5of10_vis.csv'
orgv.save_to_visualize(sg, visfilename, nodesfilename)
print('-------------------')
#multidimensional_hierarchy(5)

#get_org_success_probs('study_output/train_resultfiles.txt', 'study_output/train_dsps_multidim.json')
#get_org_success_probs('study_output/test_resultfiles.txt', 'study_output/test_dsps_multidim.json')
