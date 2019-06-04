import org.syn_hierarchy as orgh
import org.graph as orgg
import org.cluster as orgc
import org.cloud as orgk
import org.syn_fix as orgf
import numpy as np
import json
import copy
import datetime


keys = []
vecs = np.array([])
domains = []
tagdomains = dict()
domainclouds = dict()
domaintags = dict()

domainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_domains.json'
tagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_tags.json'
vecsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_vecs.json'
tagdomainsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_tagdomains.json'
domaintagsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_domaintags.json'
tagdomsimfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/tag_domain_sims.json'
tagdomtransprobsfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/tag_dom_trans_probs.json'
simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/allpair_sims.json'


def init():
    global keys, vecs, domains, tagdomains, domaintags
    print("Loading domains")
    domains = json.load(open(domainsfile))
    keys = json.load(open(tagsfile))
    vecs = np.array(json.load(open(vecsfile)))
    tagdomains = json.load(open(tagdomainsfile))
    domaintags = json.load(open(domaintagsfile))
    print('number of domains: %d domaintags: %d tagdomains: %d tags: %d vecs: %d' % (len(domains), len(domaintags), len(tagdomains), len(keys), len(vecs)))

    orgh.get_tag_domain_sim(domains, keys, vecs, tagdomsimfile)
    orgh.get_tag_domain_trans_probs(tagdomsimfile, tagdomains,          tagdomtransprobsfile)

    return keys, vecs, tagdomains


def init_plus():
    global domainclouds
    # finding the cloud
    print('cloud thteshold: 0.75')
    domainclouds = orgk.make_cloud(simfile, 0.75)
    print('domain clouds is %d' % len(domainclouds))


def singledimensional_hierarchy():
    global keys, vecs, domains, tagdomains
    print('single dim hierarchy')
    gp, success_probs = agg_fuzzy()
    print('done agg clustering and evaluating')
    init_success_probs = copy.deepcopy(success_probs)
    before_sp = sum(list(init_success_probs.values()))/float(len(init_success_probs))
    print('before_sp: %f' % before_sp)
    before_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/initial_sps_'+str(len(domains))+'.json'
    json.dump(success_probs, open(before_json, 'w'))
    print('printed to %s' % before_json)


    print('fixing')
    after_sps, fg = fix(gp)
    after_sp = sum(list(after_sps.values()))/float(len(after_sps))
    print('success prob before fix %f after fix %f.' % (before_sp, after_sp))
    after_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/final_sps_'+str(len(domains))+'.json'
    json.dump(after_sps, open(after_json, 'w'))
    print('printed to %s' % after_json)



def fix(g):
    print('fix')
    start_time = datetime.datetime.now()
    h, stats, iteration_ls, max_success_probs,                max_domain_success_probs = orgf.fix_plus(g.copy(), domains, tagdomains, domainclouds, 'taxonomy', domaintags)
    print('fix hierarchy time: %d' % (int((datetime.datetime.now()-start_time).total_seconds()*1000)))

    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(h.copy(), domains, tagdomains, domainclouds,  'taxonomy', domaintags)

    return success_probs, h



def agg_fuzzy():
    print('agg_fuzzy')
    start_time = datetime.datetime.now()
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    print('initial hierarchy time: %d' % (int((datetime.datetime.now()-start_time).total_seconds()*1000)))
    orgh.get_state_domain_sims(gp, tagdomsimfile, domains)
    orgh.init(gp, domains, simfile, tagdomtransprobsfile)
    max_success, gp, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(gp.copy(), domains, tagdomains, domainclouds, 'taxonomy', domaintags)
    avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))
    print('fuzzy: %f' % avg_success_prob)

    return gp, success_probs


init()

#orgk.all_pair_sim(domains, simfile)

init_plus()

singledimensional_hierarchy()

