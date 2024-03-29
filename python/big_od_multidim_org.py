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

domainsfile = ''
repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile, dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile = '', '', '', '', '', '', '', '', '', ''

simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json'
dimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_dims.json'
traindomainfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/od_sample_domain_names.json'
tagdomsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_domain_sims.json'
tagrepsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_rep_sims.json'
tagdomtransprobsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/tag_dom_trans_probs.json'
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
    json.dump(reps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/reps_kmeans_'+str(dim_id)+'.json', 'w'))
    json.dump(repdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/repdomains_kmeans_'+str(dim_id)+'.json', 'w'))



def get_reps_plus():
    global repdomains, reps
    for i in range(len(keys)):
        reps.append({'name': keys[i], 'mean': (vecs[i]).tolist()})
    repdomains = copy.deepcopy(tagdomains)
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

    print('dimdomains: %d, dimdomaintags: %d, dimtables: %d' % (len(dimdomains), len(dimdomaintags), len(atbs)))
    return

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

    #num_rep = max(len(ks), int(len(domains)/10.0))
    num_rep = min(len(ks), int(len(domains)/10.0))
    get_reps(num_rep, dim_id)
    #get_reps()

    # saving all indices and maps to file
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




def read_dim(dim_id):
    print('dim_id: %d' % dim_id)
    global keys, vecs, reps, repdomains, domainsfile, repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile, dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    print('assigning file names')

    domainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomains_'+str(dim_id)+'.json'
    tagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traintagdomains_'+str(dim_id)+'.json'
    domaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomainclouds_'+str(dim_id)+'.json'
    domaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomaintags_'+str(dim_id)+'.json'
    dimdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomains_'+str(dim_id)+'.json'
    dimtagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimtagdomains_'+str(dim_id)+'.json'
    dimdomaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomainclouds_'+str(dim_id)+'.json'
    dimdomaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/dimdomaintags_'+str(dim_id)+'.json'
    repsfile =  repdomainsfile

    repsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/reps_'+          str(dim_id)+'.json'
    repdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/repdomains_'+str(dim_id)+'.json'

    keys = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagnames_'+str(dim_id)+'.json', 'r'))
    vecs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagvecs_'+str(dim_id)+'.json', 'r'))
    vecs = np.array([np.array(v) for v in vecs])

    reps = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/reps_'+str(dim_id)+'.json', 'r'))
    repdomains = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/repdomains_'+str(dim_id)+'.json', 'r'))

    print('number of reps: %d' % len(reps))




def read_flat():
    global keys, vecs, domainsfile, domaincloudsfile, domaintagsfile, tagdomainsfile

    print('assigning file names')

    domainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomains.json'
    tagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traintagdomains.json'
    domaincloudsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomainclouds.json'
    domaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomaintags.json'

    keys = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagnames.json', 'r'))
    vecs = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagvecs.json', 'r'))
    vecs = np.array([np.array(v) for v in vecs])

    print('done')





def init_flat():
    print('init_flat')

    sample_domain_names = json.load(open(traindomainfile, 'r'))
    print('sample_domain_names: %d' % len(sample_domain_names))
    train_domain_names = dict()
    for s in sample_domain_names:
        train_domain_names[s] = True

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

    seen = []
    atagdomains = dict()
    adomains = []
    adomaintags = dict()
    for dom in alldomains:
        # remove the condition for all repo experiments
        #if dom['name'] not in train_domain_names:
        #    continue

        if dom['tag'] not in ks:
            continue

        if dom['name'] not in seen:
            seen.append(dom['name'])
            adomains.append(dom)

        if dom['name'] not in adomaintags:
            adomaintags[dom['name']] = []
        adomaintags[dom['name']].append(dom['tag'])

        if dom['tag'] not in atagdomains:
            atagdomains[dom['tag']] = []
        atagdomains[dom['tag']].append(dom['name'])

    print('num of alli unique domains: %d all domains: %d all tagdomains: %d' % (len(seen), len(alldomains), len(atagdomains)))
    print('adomaintags: %d' % len(adomaintags))
    alldomainclouds = orgk.make_cloud(simfile, 0.9)
    orgk.plot(alldomainclouds)
    #alldomainclouds = orgk.make_cloud(simfile, 0.75)
    print('all domainclouds: %d' % len(alldomainclouds))
    return

    #for dom in train_domain_names:
    #    domainclouds[dom] = dict()
    #    for cd, cp in alldomainclouds[dom].items():
    #        if cd in train_domain_names:
    #            domainclouds[dom][cd] = cp
    # create the cloud for all domains
    for dom in seen:
        domainclouds[dom] = dict()
        for cd, cp in alldomainclouds[dom].items():
            domainclouds[dom][cd] = cp


    print('saving train data to files.')
    json.dump(domainclouds, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomainclouds.json', 'w'))
    json.dump(adomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomains.json', 'w'))
    json.dump(atagdomains, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traintagdomains.json', 'w'))
    json.dump(adomaintags, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/traindomaintags.json', 'w'))


    json.dump(keys, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagnames.json', 'w'))
    json.dump([list(v) for v in vecs], open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_tmp/tagvecs.json', 'w'))

    orgh.get_tag_domain_trans_probs(tagdomsimfile, atagdomains, tagdomtransprobsfile)






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

    tbs = dict()
    seen = []
    atagdomains = dict()
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue

        if dom['name'] not in seen:
            seen.append(dom['name'])

        table = dom['name'][:dom['name'].rfind('_')]
        tbs[table]= True

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
    stagdomains, sdomains = orgs.stratified_sample(tagdomains, 0.45)
    print('sampled domains: %d sampled tagdomains: %d' % (len(sdomains), len(stagdomains)))
    tagdomains = copy.deepcopy(stagdomains)
    domains = []
    # making domains unique
    seen = dict()
    tbs = dict()
    sample_domain_names = []
    for domain in sdomains:
        domainname = domain['name']
        sample_domain_names.append(domainname)
        table = domainname[:domainname.rfind('_')]
        tbs[table]= True
        if domainname not in seen:
            domains.append(domain)
            seen[domainname] = True
    sample_domain_names = list(set(sample_domain_names))
    print('unique sample_domain_names: %d' % len(sample_domain_names))
    print('sampled tables: %d' % len(tbs))
    print('sampled domains: %d  unique domains: %d keys: %d vecs: %d sampled tagdomains: %d domaintags: %d' % (len(sdomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))
    print('writing train domain names')
    json.dump(sample_domain_names, open(traindomainfile, 'w'))
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

    h, iteration_sps, iteration_ls, sps, dsps = orgf.fix_plus(g, domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile, repsfile, repdomainsfile)


    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(h.copy(), domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile)

    print('success prob for train domains after fix: %f(%f)' % (max_success, sum(list(success_probs.values()))/len(success_probs)))


    return success_probs, h, domain_success_probs



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

    json.dump(domain_success_probs, open('od_output/agg_' + suffix1 + '_' + str(len(domains)) + '_domain_sps.json', 'w'))
    json.dump(success_probs, open('od_output/agg_' + suffix1 + '_' + str(len(domains)) + '_table_sps.json', 'w'))
    print('initial dsps to %s' % ('od_output/agg_' + suffix1 + '_' + str(len(domains)) + '_domain_sps.json'))

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
    after_sps, fg, after_dsps = fix(gp, 'agg_'+str(dim_id)+'of'+str(dim_num), dim_id)
    sp = sum(list(after_sps.values()))/float(len(after_sps))

    print('time to build dim %d is %d' % (dim_id, int((datetime.datetime.now()-ds).total_seconds() * 1000)))


    # saving the org
    print('sp of dim %d for domains after fix is %f.' % (dim_id, sp))
    orgfilname = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/org'+str(dim_id)+'of'+str(dim_num)+'txt'
    print('saving org to %s' % orgfilname)
    orgh.save(fg, orgfilname)

    # evaluating test domains: domains that are not in the sample
    # need to call init() to reindex all domains on name
    orgh.extend_node_dom_sims(fg, dimdomainsfile)

    all_tagdomtransprobsfile = tagdomtransprobsfile.replace('.json', '_all'+             str(dim_id)+'.json')
    orgh.init(fg, dimdomainsfile, simfile, all_tagdomtransprobsfile)


    test_success, gp, test_sps, test_likelihood, test_dsps = orgh.get_success_prob_fuzzy(fg, dimdomainsfile, dimtagdomainsfile, dimdomaincloudsfile, 'opendata', dimdomaintagsfile)
    print('success prob of dim %d for all domains: %f' % (dim_id, test_success))

    # saving domain success probs of train and test
    json.dump(after_sps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/fix_train_probs_' + str(dim_id) + '.json', 'w'))
    json.dump(after_dsps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/fix_train_domain_probs_' + str(dim_id) + '.json', 'w'))
    print('printed after: %s and %s' % ('fix_train_probs_' + str(dim_id) + '.json', 'fix_train_domain_probs_' + str(dim_id) + '.json'))

    json.dump(before_sps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/agg_train_probs_' + str(dim_id) + '.json', 'w'))
    json.dump(before_dsps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/agg_train_domain_probs_' + str(dim_id) + '.json', 'w'))
    print('printed before: %s and %s' % ('agg_train_probs_' + str(dim_id) + '.json', 'agg_train_domain_probs_' + str(dim_id) + '.json'))


    json.dump(test_dsps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/all_domain_probs_' + str(dim_id) + '.json', 'w'))
    json.dump(test_sps, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/all_table_probs_' + str(dim_id) + '.json', 'w'))
    print('printed test: %s and %s' % ('all_domain_probs_' + str(dim_id) + '.json', 'all_table_probs_' + str(dim_id) + '.json'))


    print('success prob of dim %d before fix for train domains: %f' % (dim_id, (sum(list(before_sps.values()))/len(before_sps))))

    print('success prob of dim %d after fix with %d reps: %f' % (dim_id, len(reps), (sum(list(after_sps.values()))/len(after_sps))))

    print('success prob of dim %d for all domains after fix: %f' % (dim_id, (sum(list(test_sps.values()))/len(test_sps))))

    print('end time:')
    print(datetime.datetime.now())


def flat():
    print('flat')
    global domainsfile, repsfile, repdomainsfile, dimdomaincloudsfile, dimdomaintagsfile, dimtagdomainsfile,dimdomainsfile, domainsfile, tagdomainsfile, domaincloudsfile, domaintagsfile

    gp = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs, keys)
    print('initial hierarchy with %d nodes' % len(list(gp.nodes)))
    orgh.get_state_domain_sims(gp, tagdomsimfile, domainsfile)

    orgh.init(gp, domainsfile, simfile, tagdomtransprobsfile)


    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(gp.copy(), domainsfile, tagdomainsfile, domaincloudsfile, 'opendata', domaintagsfile)

    print('the success prob of the flat org: %f' % max_success)

    #json.dump(domain_success_probs, open('od_output/flat_' + str(len(domains)) + '_domain_sps.json', 'w'))
    #json.dump(success_probs, open('od_output/flat_' + str(len(domains)) + '_table_sps.json', 'w'))
    #print('flat dsps to %s' % ('od_output/flat_' + str(len(domains)) + '_domain_sps.json'))
    #print('flat sps to %s' % ('od_output/flat_' + str(len(domains)) + '_table_sps.json'))

    json.dump(domain_success_probs, open('od_output/all_flat_' + str(len(domains)) + '_domain_sps.json', 'w'))
    json.dump(success_probs, open('od_output/all_flat_' + str(len(domains)) + '_table_sps.json', 'w'))
    print('flat dsps to %s' % ('od_output/all_flat_' + str(len(domains)) + '_domain_sps.json'))
    print('flat sps to %s' % ('od_output/all_flat_' + str(len(domains)) + '_table_sps.json'))



    return gp, success_probs, domain_success_probs



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
#TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/boosted_socrata_label_embs'
#DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/boosted_socrata_domain_embs'

init_flat()
#read_flat()
#flat()

#init()

#init_dim(0)
#load_dim(6)
#read_dim(9)
print('-------------------')
#multidimensional_hierarchy(9)

#get_org_success_probs('od_output/train_resultfiles.txt', 'od_output/train_dsps_multidim.json')
#get_org_success_probs('od_output/test_resultfiles.txt', 'od_output/test_dsps_multidim.json')
