import org.hierarchy as orgh
import org.graph as orgg
import org.plot.dist_plot as orgp
import org.cluster as orgc
import org.cloud as orgk
import org.load as orgl
import org.fix as orgf
import org.semantic as orgm
import operator
import numpy as np
import json
import copy
import os
import ntpath


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
    print('cloud thteshold: 0.75')
    domainclouds = orgk.make_cloud(simfile, 0.75)
    orgk.plot(domainclouds)


def singledimensional_hierarchy():
    print('singledimensional_hierarchy')
    global keys, vecs, domains, tagdomains
    print('single dim hierarchy')
    gp, results = agg_fuzzy('fuzzy_g10t752opint', 'strict_g10t752opint')
    print('done agg clustering and evaluating')
    init_success_probs = results['success_probs']
    before_sp = sum(list(init_success_probs.values()))/float(len(init_success_probs))
    print('before_sp: %f' % before_sp)

    print('fixing')
    sps, fg = fix(gp, 'agg_singledim_g10t752opint')
    after_sp = sum(list(sps.values()))/float(len(sps))
    print('success prob before fix %f after fix %f.' % (before_sp, after_sp))

    single_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/agg_dists_' + str(len(domains)) + '_single_g10t752opint.json'
    json.dump(sps, open(single_json, 'w'))
    print('printed to %s' % single_json)

    print('saving fg')
    graph_file = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/hierarchy_' + str(len(domains)) + '.txt'
    print('getting semantics')
    orgh.save(fg, graph_file)
    orgm.get_states_semantic(graph_file)
    print('extracting semantics of org')
    dirpath = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_sem'
    h, t = ntpath.split(graph_file)
    orgm.get_org_semantic(graph_file, os.path.join(dirpath, t))



def fix(g, hierarchy_name):
    print('fix')
    h, iteration_sps, iteration_ls = orgf.fix_plus(g, domains, tagdomains, domainclouds, 'synthetic')

    #print('strict')
    #results = orgh.evaluate(h.copy(), domains, tagdomains)
    print('fuzzy: ')
    results = orgh.fuzzy_evaluate(h.copy(), domains, tagdomains, domainclouds, 'synthetic')
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

    return success_probs, h



def agg_fuzzy(suffix1, suffix2):
    print('agg_fuzzy')
    gp = orgh.add_node_vecs(orgg.cluster_to_graph(orgc.basic_clustering(vecs, 2, 'ward', 'euclidean'), vecs, keys), vecs, keys)
    orgh.init(gp, domains, tagdomains, simfile)
    results = orgh.fuzzy_evaluate(gp.copy(), domains, tagdomains, domainclouds, 'synthetic')
    success_probs = results['success_probs']
    avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))
    print('fuzzy: %f' % avg_success_prob)

    print('number of tables: %d' % len(list(results['success_probs'].values())))

    json.dump(results['success_probs'], open('synthetic_output/agg_dists' + str(len(domains)) + suffix1 + '.json', 'w'))
    print('printed the fuzzy eval of agg hierarchy to %s.' % ('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/org/plot/agg_' + str(len(domains)) + suffix1 + '.pdf'))

    return gp, results

def flat(suffix):
    print('flat')
    g = orgh.add_node_vecs(orgg.get_flat_cluster_graph(keys), vecs)
    orgh.init(g, domains, tagdomains)
    #results = orgh.evaluate(g, domains, tagdomains)
    results = orgh.fuzzy_evaluate(g.copy(), domains, tagdomains, domainclouds, 'synthetic')
    print(results['success_probs'])
    jsonfile = 'synthetic_output/flat_dists_' + str(len(domains)) + '_' + suffix + '.json'
    json.dump(results['success_probs'], open(jsonfile, 'w'))
    print('number of tables: %d' % len(results['success_probs']))
    print('sp: %f' % (sum(list(results['success_probs'].values()))/len(results['success_probs'])))
    print('printed to %s' % jsonfile)



init(20)

simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
#orgk.all_pair_sim(domains, simfile)

init_plus()

singledimensional_hierarchy()

#flat('g10')

#gp, results = agg_fuzzy('fuzzy_gamma10', 'strict')

#orgc.basic_plus(vecs, 5, 'ward', 'euclidean')
