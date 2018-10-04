import org.hierarchy as orgh
import org.graph as orgg
import org.plot.dist_plot as orgp
import org.cluster as orgc
import org.sample as orgs
import org.cloud as orgk
import org.fix as orgf
import org.semantic as orgm
#import org.tag as orgt
#import importlib
#import operator
import csv
import numpy as np
import json
#import networkx as nx
import copy
import datetime
import ntpath
import os


keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
domainclouds = dict()
domaintags = dict()
comptestdomains = []
comptagdomains = dict()


def write_emb_header():
    EMBS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_embs'
    fasttextSize = 300
    sf = open(EMBS_FILE, 'a', newline="\n")
    header = ['dataset_name', 'column_id', 'column_name', 'column_type']
    for i in range(fasttextSize):
        header.append('f' + str(i))
    swriter = csv.writer(sf, delimiter=',', escapechar='\\', lineterminator='\n', quoting=csv.QUOTE_NONE)
    swriter.writerow(header)
    sf.close()



def init(tag_num):
    global keys, vecs, domains, tagdomains, domaintags, comptagdomains, comptestdomains
    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        if t.startswith('socrata_'):
            ks.append(t)
            vs.append(e)


    tag_num = len(ks)

    print('tag_embs: %d ks: %d vs: %d' % (len(tag_embs), len(vs), len(vs)))

    keys = ks
    vecs = np.array(vs)

    alldomains = json.load(open(DOMAIN_FILE))

    tbs = []

    adomaintags = dict()
    atagdomains = dict()
    for dom in alldomains:

        if dom['tag'] not in ks:
            continue

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

    print('num of tables: %d' % len(tbs))
    print('all domains: %d atagdomains: %d' % (len(alldomains), len(atagdomains)))

    # choosing the most frequent tags
    #atagdomcounts = {t:len(ds) for t, ds in atagdomains.items()}
    #satagdomcounts = sorted(atagdomcounts.items(), key=operator.itemgetter(1), reverse=True)
    #tvs = [p[0] for p in satagdomcounts]
    #tvs = tvs[:tag_num]

    # random sample of tags
    tvs = list(atagdomains.keys())[:tag_num]
    print('tvs: %d' % len(tvs))
    tagdomains = dict()
    domains = []
    lvecs = []
    keys = []
    for t in tvs:
        keys.append(t)
        lvecs.append(tag_embs[t])
        tagdomains[t] = copy.deepcopy(atagdomains[t])
        domains.extend(list(tagdomains[t]))
    vecs = np.array(lvecs)
    print('domains: %d vecs: %d keys: %d' % (len(domains), len(vecs), len(keys)))
    comptestdomains = list(domains)
    comptagdomains = list(tagdomains)
    # sampling
    stagdomains, sdomains = orgs.stratified_sample(tagdomains, 0.6)
    print('sampled domains: %d sampled tagdomains: %d' % (len(sdomains), len(stagdomains)))
    print('testdomains: %d' % len(comptestdomains))
    tagdomains = copy.deepcopy(stagdomains)
    domaintags = copy.deepcopy(adomaintags)
    domains = []
    # making domains unique
    seen = dict()
    tbs = []
    for domain in sdomains:
        if domain['name'] not in seen:
            domains.append(domain)
            seen[domain['name']] = True
            table = domain['name'][:domain['name'].rfind('_')]
            if table not in tbs:
                tbs.append(table)
    print('sampled tables: %d' % len(tbs))
    print('sdomains: %d  domains: %d keys: %d vecs: %d tagdomains: %d domaintags: %d' % (len(sdomains), len(domains), len(keys), len(vecs), len(tagdomains), len(domaintags)))


def init_load(simfile):
    global keys, vecs, domains, tagdomains

    # loading samples from file
    tables = []
    ldomains = []
    sims = json.load(open(simfile, 'r'))
    for d1, d2sims in sims.items():
        ldomains.append(d1)
        t = d1[:d1.rfind('_')]
        if t not in tables:
            tables.append(t)
    print('number of tables: %d domains: %d' % (len(tables), len(ldomains)))

    vs = []
    tag_embs = json.load(open(TAG_EMB_FILE))
    for t, e in tag_embs.items():
        keys.append(t)
        vs.append(e)
    vecs = np.array(list(vs))
    print('initial tags: %d vs: %d' % (len(keys), len(vs)))

    alldomains = json.load(open(DOMAIN_FILE))
    seen = []
    for dom in alldomains:
        if dom['name'] not in ldomains:
            continue
        if dom['tag'] not in keys:
            continue
        if dom['tag'] not in tagdomains:
            tagdomains[dom['tag']] = []
        tagdomains[dom['tag']].append(dom)
        if dom['name'] not in seen:
            domains.append(dom)

    print('domains: %d all domains: %d tagdomains: %d' % (len(domains), len(alldomains), len(tagdomains)))




def init_plus():
    global domainclouds
    # finding the cloud
    #simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims_' + str(len(domains)) + '.json'
    domainclouds = orgk.make_cloud(simfile, 0.75)
    print('domainclouds: %d' % len(domainclouds))
    orgk.plot(domainclouds)


def fix(g, hierarchy_name):
    print('fix')
    h, iteration_sps, iteration_ls = orgf.fix_plus(g, domains, tagdomains, domainclouds, 'opendata', domaintags)

    #print('strict')
    #results = orgh.evaluate(h.copy(), domains, tagdomains)
    print('fuzzy: ')
    results = orgh.fuzzy_evaluate(h.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)
    success_probs = results['success_probs']

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

    return success_probs, h



def agg_fuzzy(suffix1, suffix2):
    print('agg_fuzzy')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    print('done clustering')
    print('initial hierarchy with %d nodes' % len(list(gp.nodes)))
    orgh.init(gp, domains, tagdomains, simfile)
    results = orgh.fuzzy_evaluate(gp.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)

    #json.dump(results['success_probs'], open('od_output/agg_dists' + str(len(domains)) + suffix1 + '.json', 'w'))
    #print('printed the fuzzy eval of agg hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf'))

    return gp, results['success_probs']


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

    org_filenames = []

    for i, ts in dims.items():

        print('dim %d' % i)
        ds = datetime.datetime.now()
        keys = ts
        vecs = []
        domains = []
        domainclouds = dict()
        domain_names = {}
        tagdomains = dict()
        print('alldomainclouds: %d' % len(alldomainclouds))
        for t in ts:
            vecs.append(allvecs[allkeys.index(t)])
            for td in alltagdomains[t]:
                if td['name'] not in domain_names:
                    domains.append(td)
                    domain_names[td['name']] = True
            tagdomains[t] = list(alltagdomains[t])
        vecs = np.array(vecs)
        # finding test domains
        testdomains = []
        for t in ts:
            for pd in comptagdomains.item[t]:
                if pd['name'] not in domain_names:
                    testdomains.append(pd)

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
                success_probs_before_intersect[t] = (1.0-p)
            success_probs_before[t] += p
            success_probs_before_intersect[t] *= (1.0-p)

        de = datetime.datetime.now()
        delapsed = de - ds
        print('elapsed time of dim %d is %d' % (i, int(delapsed.total_seconds() * 1000)))

        print('fixing cluster %d' % i)
        sps, fg = fix(gp, 'agg_'+str(i)+'f2op')
        sp = sum(list(sps.values()))/float(len(sps))
        print('sp of dim %d after fix is %f.' % (i, sp))
        for t, p in sps.items():
            if t not in success_probs_after:
                success_probs_after[t] = 0.0
                success_probs_after_intersect[t] = (1.0-p)
            success_probs_after[t] += p
            success_probs_after_intersect[t] *= (1.0-p)

        # results = orgh.fuzzy_evaluate(g.copy(), domains, tagdomains, domainclouds,           'opendata', domaintags)
        print('saving fg')
        graph_file = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/hierarchy_' + str(len(domains)) + '_' + str(i) + '.txt'
        org_filenames.append(graph_file)
        print('getting semantics')
        orgh.save(fg, graph_file)
        orgm.init_fromfile('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_embs')
        orgm.get_states_semantic(graph_file)
        print('---------------')

    for t, p in success_probs_before.items():
        success_probs_before[t] = 1-success_probs_before_intersect[t]
        #if success_probs_before[t] != success_probs_before_intersect[t]:
        #    success_probs_before[t] = (p-success_probs_before_intersect[t])
        if success_probs_before[t] > 1.0:
            success_probs_before[t] = 1.0
    for t, p in success_probs_after.items():
        success_probs_after[t] =1.0-success_probs_after_intersect[t]
        #if success_probs_after[t] != success_probs_after_intersect[t]:
        #   success_probs_after[t] = (p-success_probs_after_intersect[t])
        if success_probs_after[t] > 1.0:
            success_probs_after[t] = 1.0


    before_sp = sum(list(success_probs_before.values()))/float(len(success_probs_before))
    print('success prob of multidimensions before fix: %f' % before_sp)


    after_sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of multidimensions after fix: %f' % after_sp)

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/multidim_dists_before_' + str(len(alldomains)) + '_' + str(dim_num) + '.json'
    json.dump(success_probs_before, open(multi_json, 'w'))
    print('printed mult results to %s.' % multi_json)

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/multidim_dists_' + str(len(alldomains)) + '_' + str(dim_num) + '_g10rhap.json'
    json.dump(success_probs_after, open(multi_json, 'w'))
    print('printed mult results to %s.' % multi_json)


    print('extracting semantics of orgs')
    dirpath = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_sem'
    for of in org_filenames:
        h, t = ntpath.split(of)
        orgm.get_org_semantic(of, os.path.join(dirpath, t))


def flat(suffix):
    print('flat')
    g = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs, keys)
    orgh.init(g, domains, tagdomains, simfile)
    results = orgh.fuzzy_evaluate(g.copy(), domains, tagdomains, domainclouds, 'opendata', domaintags)
    jsonfile = 'od_output/flat_dists_' + str(len(domains)) + '_' + suffix + '.json'
    json.dump(results['success_probs'], open(jsonfile, 'w'))
    print('number of tables: %d' % len(results['success_probs']))
    print('sp: %f' % (sum(list(results['success_probs'].values()))/len(results['success_probs'])))
    print('printed to %s' % jsonfile)



#orgt.get_good_labels()

#write_emb_header()

#orgt.tag_embs()

# cat /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-30k.list | xargs -d '\n' -P 20 -n 1 python python/samples/json2emb.py

TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_embs'
DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/domain_embs'

init(1000)

simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims_' + str(len(domains)) + '.json'
orgk.all_pair_sim(domains, simfile)
#
#init_load(simfile)

init_plus()

multidimensional_hierarchy(2)

#flat('od')

#orgc.cmeans_clustering(keys, vecs)
#dims = orgh.get_dimensions(keys, vecs, 10)
#dims = orgh.get_dimensions_plus(keys, vecs)

#gf1 = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/hierarchy_1355_1.txt'
#gf2 = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/hierarchy_105_0.txt'
#org_filenames = [gf1, gf2]
#print('extracting semantics of orgs')
#dirpath = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_sem'
#orgm.init_fromfile('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_embs')
#for of in org_filenames:
#    h, t = ntpath.split(of)
#    orgm.get_org_semantic(of, os.path.join(dirpath, t))

