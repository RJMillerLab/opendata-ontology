import org.syn_hierarchy as orgh
import org.cloud as orgk
import org.graph as orgg
import numpy as np
import json
import networkx as nx


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
taxonomyfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_taxonomy.json'
#taxonomyfile = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/kbtaxonomy/yagocloud_data/agri_taxonomy_trim.json'

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


def init_plus():
    global domainclouds
    # finding the cloud
    print('cloud thteshold: 0.75')
    domainclouds = orgk.make_cloud(simfile, 0.75)


def evaluate_taxonomy():
    global keys, vecs, domains, tagdomains
    print('single dim hierarchy')
    h = load_taxonomy()
    vecs = np.array(vecs)
    g = orgh.add_node_vecs(h, vecs, keys)
    orgh.get_state_domain_sims(g, tagdomsimfile, domains)
    orgh.init(g, domains, simfile, tagdomtransprobsfile)
    max_success, g, success_probs, likelihood, domain_success_probs = orgh.get_success_prob_fuzzy(g.copy(), domains, tagdomains, domainclouds,    'taxonomy', domaintags)
    avg_success_prob = sum(list(success_probs.values()))/float(len(success_probs))

    print('taxonomy_sp: %f' % avg_success_prob)

    #taxonomy_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/agri_taxonomy_trim_' + str(len(domains)) + '.json'
    taxonomy_json = '/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/yagocloud_output/agri_taxonomy_' + str(len(domains)) + '.json'
    json.dump(success_probs, open(taxonomy_json, 'w'))
    ps = list(success_probs.values())
    ps.sort(reverse=True)
    print('printed to %s' % taxonomy_json)

def load_taxonomy():
    global keys
    g = nx.DiGraph()
    edges = json.load(open(taxonomyfile))
    node_tag = dict()
    tag_node = dict()
    for i in range(len(keys)):
        node_tag[i] = keys[i]
        tag_node[keys[i]] = i
    print('tag_node: %d node_tag: %d' % (len(tag_node), len(node_tag)))
    for e in edges:
        if e[0] in keys:
            print('sth wrong %s' % e[0])
        e0, e1 = 0, 0
        if e[0] in tag_node:
            e0 = tag_node[e[0]]
        else:
            e0 = len(tag_node)
            tag_node[e[0]] = e0
            node_tag[e0] = e[0]
        if e[1] in tag_node:
            e1 = tag_node[e[1]]
        else:
            e1 = len(tag_node)
            tag_node[e[1]] = e1
            node_tag[e1] = e[1]
        g.add_edge(e0,e1)
    print('num nodes: %d' % len(g.nodes))
    print('num edges: %d' % len(g.edges))
    if max(list(orgg.get_leaves(g))) >= len(keys):
        print('leaves and tags different')
    for n in g.nodes:
        g.node[n]['tag'] = node_tag[n]
    orgg.height(g)
    orgg.branching_factor(g)
    return g


init()

#orgk.all_pair_sim(domains, simfile)

init_plus()

evaluate_taxonomy()

