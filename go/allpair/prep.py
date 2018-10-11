import json

domains = json.load(open("/home/fnargesian/FINDOPENDATA_DATASETS/10k/socrata_domain_embs", 'r'))
domainsmap = dict()
for dom in domains:
    domainsmap[dom['name']] = dom['mean']
json.dump(domainsmap, open("/home/fnargesian/FINDOPENDATA_DATASETS/socrata/socrata_domain_map_embs", 'w'))
