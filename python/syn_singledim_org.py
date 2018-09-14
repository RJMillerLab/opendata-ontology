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


def flat(suffix):
    print('flat')
    g = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs)
    orgh.init(g, domains, tagdomains)
    #results = orgh.evaluate(g, domains, tagdomains)
    results = orgh.fuzzy_evaluate(g.copy(), domains, tagdomains, domainclouds)
    tag_dists = results['success_probs']
    success_probs = dict()
    for t, p in tag_dists.items():
        #success_probs[t] = p * (1.0/len(tag_dists))
        success_probs[t] = p
    json.dump(success_probs, open('synthetic_output/flat_dists_' + str(len(domains)) + suffix + '.json', 'w'))
    print('printed to %s' % ('flat_dists_' +  str(len(domains)) + suffix + '.pdf'))


def singledimensional_hierarchy():
    print('singledimensional_hierarchy')
    global keys, vecs, domains, tagdomains
    print('single dim hierarchy')
    gp, results = agg_fuzzy('fuzzy_f1op', 'strict_f1op')
    print('done agg clustering and evaluating')
    init_success_probs = results['success_probs']
    before_sp = sum(list(init_success_probs.values()))/float(len(init_success_probs))
    print('before_sp: %f' % before_sp)

    print('fixing')
    sps = fix(gp, 'agg_singledim_f1op')
    after_sp = sum(list(sps.values()))/float(len(sps))
    print('success prob before fix %f after fix %f.' % (before_sp, after_sp))

    single_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists_' + str(len(domains)) + '_single_f1op.json'
    json.dump(sps, open(single_json, 'w'))
    single_pdf = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + '_single_f1op.pdf'
    orgp.plot(single_json, single_pdf, 'single-dimensional Hierarchy - ' + str(before_sp) + '->' + str(after_sp))




def fix(g, hierarchy_name):
    print('fix')
    h, iteration_sps, iteration_ls = orgf.fix_plus(g, domains, tagdomains, domainclouds)

    print('strict')
    results = orgh.evaluate(h.copy(), domains, tagdomains)
    print('fuzzy: ')
    results = orgh.fuzzy_evaluate(h.copy(), domains, tagdomains, domainclouds)
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
    results = orgh.fuzzy_evaluate(gp.copy(), domains, tagdomains, domainclouds)
    success_probs = results['success_probs']
    avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))
    print('fuzzy: %f' % avg_success_prob)

    #results = orgh.evaluate(gp.copy(), domains, tagdomains)
    #success_probs = results['success_probs']
    #avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))
    #print('strict: %f' % avg_success_prob)

    json.dump(results['success_probs'], open('synthetic_output/agg_dists' + str(len(domains)) + suffix1 + '.json', 'w'))
    print('printed the fuzzy eval of agg hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf'))

    return gp, results


def agg(suffix):
    print('agg')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs)
    orgh.init(gp, domains, tagdomains)
    results = orgh.evaluate(gp, domains, tagdomains)

    json.dump(results['success_probs'], open('synthetic_output/agg_dists' + str(len(domains)) + suffix + '.json', 'w'))
    json.dump(results['tag_ranks'], open('synthetic_output/agg_ranks' + str(len(domains)) + suffix + '.json', 'w'))
    orgp.plot('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists' + str(len(domains)) + suffix + '.json', '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix + '.pdf', 'Agglomerative Clustering')
    print('printed the initial hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix + '.pdf'))




init(500)

#simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
#orgk.all_pair_sim(domains, simfile)

init_plus()

#flat('flat_br')

singledimensional_hierarchy()

#agg_fuzzy('fuzzy', 'strict')

