import emb
import gmi
import numpy as np
import os

DATA_DIR = 'data'

def get_class_emb(class_name):
    ''' reads teh list of entities of an ontology class and generates
        the emb vector for each entity
    '''
    with open(os.path.join(DATA_DIR, class_name), 'r') as cf:
        entities = cf.read().splitlines()
    return emb.get_features(entities)

def get_distance(class1, class2):
    pv = np.cov(class1.T)
    pm = np.average(class1, axis=0)
    qv = np.cov(class2.T)
    qm = np.average(class2, axis=0)
    # not sure if sum is the right way of making d_kl scalar.
    return abs(gmi.kullback_leibler_divergence(pm, pv, qm, qv).sum())

def class_isa():
    classes = ['wikicat_Fisheries', 'wikicat_Fisheries_and_aquaculture_research_institutes',  'wikicat_Fisheries_in_Canada', 'wikicat_Provinces_and_territories_of_Canada', 'wikicat_Sustainable_fisheries','wordnet_fishery_103350880', 'wordnet_state_108654360']
    class_embs = {}
    for c in classes:
        ce = get_class_emb(c)
        if ce is not None:
            class_embs[c] = ce
        else:
            print('emb of %s is none.' %c)
    pairs = []
    dists = []
    for c1 in classes:
        for c2 in classes:
            if c1 != c2:
                if c1 in class_embs and c2 in class_embs:
                    pairs.append(c1 + '--' + c2)
                    dists.append(get_distance(class_embs[c1],class_embs[c2]))
    ids = list(np.argsort(np.asarray(dists)))
    for i in ids:
        print(pairs[i])
        print(dists[i])
        print('----------------------------------')
#rand_class = np.random.uniform(low=-1.0, high=1.0, size=(7,300))
#class_isa()
