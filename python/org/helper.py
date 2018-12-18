import json
def get_att_num():
    tps = json.load(open('../od_output/test_dsps_multidim.json', 'r'))
    import glob, os
    os.chdir('/home/fnargesian/FINDOPENDATA_DATASETS/10k/domains')
    count = 0
    for t, p in tps.items():
        os.chdir(os.path.join('/home/fnargesian/FINDOPENDATA_DATASETS/10k/domains', t))
        count += len(glob.glob("*.ft-mean"))
        print(count)
    print('tot dom')
    print(count)

def split_cluster():
    olds = json.load(open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/boosted_od_dims.bkp.json', 'r'))
    news = [[] for i in range(len(olds))]
    news.append([])
    news.append([])
    for o in olds:
        print(len(o))
    print('new')
    for i in range(len(olds)):
        if i == 8:
            news[8] = list(olds[8][:1648])
            news[10] = list(olds[8][1648:3148])
            news[11] = list(olds[8][3148:])
        else:
            news[i] = list(olds[i])

    for n in news:
        print(len(n))
    json.dump(news, open('/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/od_output/boosted_od_dims2.json', 'w'))



def repo_stats():
    lts = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json', 'r'))
    label_names = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json', 'r'))
    socrata_tables = dict()
    ckan_tables = dict()
    socrata_labels =dict()
    ckan_labels = dict()
    for t, ls in lts.items():
        for l in ls:
            if label_names[str(l)].startswith('socrata'):
                if t not in socrata_tables:
                    socrata_tables[t] = []
                if l not in socrata_tables[t]:
                    socrata_tables[t].append(l)
                if l not in socrata_labels:
                    socrata_labels[l] = []
                if t not in socrata_labels[l]:
                    socrata_labels[l].append(t)
    for t, ls in lts.items():
        if len(ckan_tables) >= len(socrata_tables):
            continue
        for l in ls:
            if label_names[str(l)].startswith('ckan'):
                if t not in ckan_tables:
                    ckan_tables[t] = []
                if l not in ckan_tables[t]:
                    ckan_tables[t].append(l)
                if l not in ckan_labels:
                    ckan_labels[l] = []
                if t not in ckan_labels[l]:
                    ckan_labels[l].append(t)

    socrata_tables_count = {t: len(ls) for t, ls in socrata_tables.items()}
    socrata_labels_count = {t: len(ls) for t, ls in socrata_labels.items()}
    ckan_tables_count = {t: len(ls) for t, ls in ckan_tables.items()}
    ckan_labels_count = {t: len(ls) for t, ls in ckan_labels.items()}

    print('ckan tables: %d' % len(ckan_tables))
    print('socrata tables: %d' % len(socrata_tables))
    print('all: %d' % len(lts))

    print('socrata min label num: %d max: %d avg: %d' % (min(list(socrata_tables_count.values())), max(list(socrata_tables_count.values())), sum(list(socrata_tables_count.values()))/float(len(socrata_tables_count))))
    print('socrata min table num: %d max: %d avg: %d' % (min(list(socrata_labels_count.values())), max(list(socrata_labels_count.values())),      sum(list(socrata_labels_count.values()))/float(len(socrata_labels_count))))
    print('ckan min label num: %d max: %d avg: %d' % (min(list(ckan_tables_count.values())), max(list(ckan_tables_count.values())),             sum(list(ckan_tables_count.values()))/float(len(ckan_tables_count))))
    print('ckan min table num: %d max: %d avg: %d' % (min(list(ckan_labels_count.values())), max(list(ckan_labels_count.values())),             sum(list(ckan_labels_count.values()))/float(len(ckan_labels_count))))



#split_cluster()

repo_stats()
