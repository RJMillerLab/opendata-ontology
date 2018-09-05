import org
import org.tag as orgt
import org.hierarchy as orgh
import org.graph as orgg
import org.plot.dist_plot as orgp
import org.cluster as orgc
import org.sample as orgs
import org.cloud as orgk
import org.fix as orgf
import importlib
import csv
import numpy as np
import json
import networkx as nx
import copy
import datetime


keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
domainclouds = dict()

def reload():
    importlib.reload(org)
    importlib.reload(orgt)


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



def init():
    global keys, vecs, domains, tagdomains
    tag_embs = json.load(open(TAG_EMB_FILE))
    ks = []
    vs = []
    for t, e in tag_embs.items():
        ks.append(t)
        vs.append(e)

    print('ks: %d vs: %d' % (len(ks), len(vs)))
    keys = list(ks[:500])
    vecs = np.array(list(vs[:500]))

    alldomains = json.load(open(DOMAIN_FILE))
    domains = []
    for dom in alldomains:
        if dom['tag'] not in keys:
            continue
        domains.append(dom)
        if dom['tag'] not in tagdomains:
            tagdomains[dom['tag']] = []
        tagdomains[dom['tag']].append(dom)
    print('domains: %d' % len(domains))
    print('tagdomains: %d' % len(tagdomains))
    print('keys: %d' % len(keys))
    print('vecs: %d' % len(vecs))
    # sampling
    stagdomains, sdomains = orgs.stratified_sample(tagdomains, 0.5)
    print('sampled domains: %d' % len(sdomains))
    print('sampled tagdomains: %d' % len(stagdomains))
    tagdomains = stagdomains
    domains = sdomains


def init_plus():
    global domainclouds
    # finding the cloud
    simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json'
    domainclouds = orgk.make_cloud(simfile, 0.9)
    orgk.plot(domainclouds)


def fix(g, hierarchy_name):
    print('fix')
    #init_results = orgh.evaluate(g.copy(), domains, tagdomains)
    init_results = orgh.fuzzy_evaluate(g.copy(), domains, tagdomains, domainclouds)
    init_success_probs = init_results['success_probs']
    s = datetime.datetime.now()
    h, iteration_sps, iteration_ls = orgf.fix_plus(g, domains, tagdomains, domainclouds)
    print('after fix: node %d edge %d' % (len(h.nodes), len(h.edges)))
    e = datetime.datetime.now()
    elapsed = e - s
    print('elapsed time: %d' % int(elapsed.total_seconds() * 1000))

    #results = orgh.evaluate(h, domains, tagdomains)
    results = orgh.fuzzy_evaluate(h, domains, tagdomains, domainclouds)
    success_probs = results['success_probs']
    print('improvement after fixing: %f' % orgh.get_improvement(init_success_probs, success_probs))

    tag_dists = results['success_probs']
    json.dump(tag_dists, open('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json', 'w'))
    json.dump(iteration_sps, open('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json', 'w'))
    json.dump(iteration_ls, open('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json', 'w'))

    fix_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_tag_dists.json'
    fix_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'
    orgp.plot(fix_json, fix_pdf, 'Fixed Organization')

    sps_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.json'
    sps_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'
    orgp.plot_fix(sps_json, sps_pdf, 'Average Success Prob', hierarchy_name)

    ls_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.json'
    ls_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'
    orgp.plot_fix(ls_json, ls_pdf, 'Log Likelihood', hierarchy_name)

    print('printed to %s' % ('fix_' + hierarchy_name + '_prob_' + str(len(domains)) + '.pdf'))
    print('printed iteration success plot to %s' % ('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_sps.pdf'))
    print('printed iteration likelihood plot to %s' % ('output/fix_' + hierarchy_name + '_' + str(len(domains)) + '_iteration_ls.pdf'))

    return success_probs



def agg_fuzzy(suffix1, suffix2):
    print('fuzzy agg')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs)
    orgh.init(gp, domains, tagdomains)

    results = orgh.fuzzy_evaluate(gp, domains, tagdomains, domainclouds)

    json.dump(results['success_probs'], open('od_output/agg_dists' + str(len(domains)) + suffix1 + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/agg_dists' + str(len(domains)) + suffix1 + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf', 'fuzzy eval - single dimension')
    print('printed the fuzzy eval of agg hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf'))

    return gp, results['success_probs']

    results = orgh.evaluate(gp, domains, tagdomains)

    json.dump(results['success_probs'], open('od_output/agg_dists' + str(len(domains)) + suffix2 + '.json', 'w'))
    json.dump(results['tag_ranks'], open('od_output/agg_ranks' + str(len(domains)) + suffix2 + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/agg_dists' + str(len(domains)) + suffix2 + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix2 + '.pdf', 'strict eval - single dimension')
    print('printed the initial hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix2 + '.pdf'))


def agg(suffix):
    print('agg')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs)
    orgh.init(gp, domains, tagdomains)
    results = orgh.evaluate(gp, domains, tagdomains)

    json.dump(results['success_probs'], open('od_output/agg_dists' + str(len(domains)) + suffix + '.json', 'w'))
    json.dump(results['tag_ranks'], open('od_output/agg_ranks' + str(len(domains)) + suffix + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/agg_dists' + str(len(domains)) + suffix + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix + '.pdf', 'Agglomerative Clustering')
    print('printed the initial hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix + '.pdf'))


def dimensional_hierarchy(dim_num):
    singledimensional_hierarchy()
    multidimensional_hierarchy(dim_num)


def singledimensional_hierarchy():
    global keys, vecs, domains, tagdomains
    print('single dim hierarchy')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs)
    print('done agg clustering')
    orgh.init(gp, domains, tagdomains)
    results = orgh.evaluate(gp, domains, tagdomains)
    print('done evaluating')
    #success_probs_agg = results['success_probs']
    #agg_sp = sum(list(success_probs_agg.values()))/float(len(success_probs_agg))

    single_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/agg_dists_' + str(len(domains)) + '_single.json'
    json.dump(results['success_probs'], open(single_json, 'w'))
    single_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + '_single.pdf'
    print('printed single dim to %s' % single_pdf)

    success_probs_after = dict()
    print('fixing')
    sps = fix(gp, 'agg_singledim')
    sp = sum(list(sps.values()))/float(len(sps))
    print('sp after fix is %f.' % sp)
    for t, p in sps.items():
        if t not in success_probs_after:
            success_probs_after[t] = 0.0
        success_probs_after[t] += p
    after_sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of multidimensions after fix: %f' % after_sp)

    #multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/fixed_singledim_dists_' + str(len(domains)) + '.json'
    #json.dump(success_probs_after, open(multi_json, 'w'))
    #multi_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/fixed_singledim_' + str(len(domains)) + '_' + '.pdf'
    #orgp.plot(multi_json, multi_pdf, 'single-dimensional Hierarchy - ' + str(before_sp) + '->' + str(after_sp))



def multidimensional_hierarchy(dim_num):
    global keys, vecs, domains, tagdomains, domainclouds
    dims = orgh.get_dimensions(keys, vecs, dim_num)
    allkeys = list(keys)
    allvecs = copy.deepcopy(vecs)
    alltagdomains = copy.deepcopy(tagdomains)
    alldomains = copy.deepcopy(domains)
    alldomainclouds = copy.deepcopy(domainclouds)
    success_probs_before = dict()
    success_probs_after = dict()
    for i, ts in dims.items():
        print('dim %d' % i)
        ds = datetime.datetime.now()
        keys = ts
        vecs = []
        domains = []
        domainclouds = dict()
        domain_names = []
        tagdomains = dict()
        for t in ts:
            vecs.append(allvecs[allkeys.index(t)])
            domains.extend(alltagdomains[t])
            tagdomains[t] = list(alltagdomains[t])
        vecs = np.array(vecs)
        for d in domains:
            domain_names.append(d['name'])
        for d1, d2s in alldomainclouds.items():
            if d1 not in domain_names:
                continue
            domainclouds[d1] = dict()
            for d2, s in d2s.items():
                if d2 in domain_names:
                    domainclouds[d1][d2] = s

        print('tags: %d' % (len(keys)))
        print('domains: %d  tagdomains: %d' % (len(domains), len(tagdomains)))
        #gp, sps = agg('agg'+str(i))
        gp, sps = agg_fuzzy('2agg'+str(i), '')
        nx.write_gpickle(gp, 'od_output/od_agg_' + str(i) + '.pickle')
        for t, p in sps.items():
            if t not in success_probs_before:
                success_probs_before[t] = 0.0
            #success_probs_before[t] += (p*(1.0/dim_num))
            success_probs_before[t] += p
            if success_probs_before[t] > 1.0:
                success_probs_before[t] = 1.0


        de = datetime.datetime.now()
        delapsed = de - ds
        print('elapsed time of dim %d is %d' % (i, int(delapsed.total_seconds() * 1000)))

        print('fixing cluster %d' % i)
        sps = fix(gp, 'agg_'+str(i))
        sp = sum(list(sps.values()))/float(len(sps))
        print('sp of dim %d after fix is %f.' % (i, sp))
        for t, p in sps.items():
            if t not in success_probs_after:
                success_probs_after[t] = 0.0
            #success_probs_after[t] += (p*(1.0/dim_num))
            success_probs_after[t] += p
        print('---------------')
    before_sp = sum(list(success_probs_before.values()))/float(len(success_probs_before))
    print('success prob of multidimensions before fix: %f' % before_sp)
    #success_probs_after = copy.deepcopy(success_probs_before)
    after_sp = sum(list(success_probs_after.values()))/float(len(success_probs_after))
    print('success prob of multidimensions after fix: %f' % after_sp)

    multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/multidim_dists_' + str(len(alldomains)) + '_' + str(dim_num) + '.json'
    json.dump(success_probs_after, open(multi_json, 'w'))
    multi_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/multidim_' + str(len(alldomains)) + '_' + str(dim_num) + '.pdf'
    orgp.plot(multi_json, multi_pdf, str(dim_num) + '-dimensional Hierarchy - ' + str(before_sp) + '->' + str(after_sp))
    #orgp.double_plot(single_json, single_pdf, 'Single Hierarchy - Agglomerative - ' + str(agg_sp), multi_json, multi_pdf, str(dim_num) + '-dimensional Hierarchy - ' + str(before_sp) + '->' + str(after_sp))


    #multi_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/multidim_dists_' + str(len(alldomains)) + '_' + str(dim_num) + '.json'
    #json.dump(success_probs_before, open(multi_json, 'w'))
    #multi_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/multidim_' + str(len(alldomains)) + '_' + str(dim_num) + '.pdf'
    #orgp.plot(multi_json, multi_pdf, str(dim_num) + '-dimensional Hierarchy - ' + str(before_sp))

    print('printed mult results to %s.' % multi_pdf)
    return success_probs_before


#orgt.get_good_labels()

#write_emb_header()

#orgt.tag_embs()

# cat /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-30k.list | xargs -d '\n' -P 20 -n 1 python python/samples/json2emb.py

TAG_EMB_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_embs'
DOMAIN_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/domain_embs'

init()

simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/allpair_sims.json'
orgk.all_pair_sim(domains, simfile)

init_plus()

#agg_fuzzy('fuzzy', 'strict')

multidimensional_hierarchy(5)



