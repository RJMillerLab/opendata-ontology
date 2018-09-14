import org.hierarchy as orgh
import org.graph as orgg
import org.cluster as orgc
import org.cloud as orgk
import org.load as orgl
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



init(500)

#simfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/synthetic_output/allpair_sims.json'
#orgk.all_pair_sim(domains, simfile)

init_plus()

#flat('flat_br')

