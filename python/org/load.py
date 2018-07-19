import os
import sqlite3
import numpy as np
import org.hierarchy as orgh

REPO_PATH = '/home/kenpu/clones/tagcloud-nlp-generator/output'
FT_SQLITE3_PATH = '/home/kenpu/clones/tagcloud-nlp-generator/ft.sqlite3'

def list_table_names():
    for f in os.listdir(REPO_PATH):
        if f.endswith("csv"):
            yield f[:-4]

def iter_domains():
    for table in list_table_names():
        csvf = os.path.join(REPO_PATH, "%s.csv" % table)
        tagf = os.path.join(REPO_PATH, "%s.tags.vec" % table)
        with open(csvf, 'r') as f:
            lines = list(f)
            rows = [x.split(',') for x in lines]
        with open(tagf, 'r') as f:
            lines = list(f)
            tags = [x.split(" ")[0] for x in lines]

        for i in range(len(tags)):
            domain = [row[i] for row in rows]
            tag = tags[i]
            yield dict(tag=tag, domain=domain, name=table+'_'+str(i))

def lookup_ft_vector(cursor, word):
    cursor.execute("select vec from wv where word = ?", [word])
    result = cursor.fetchone()
    if result:
        return np.frombuffer(result[0], dtype='float32')
    else:
        return None

def add_ft_vectors(domains):
    db = sqlite3.connect(FT_SQLITE3_PATH)
    cursor = db.cursor()
    for dom in domains:
        vecs = []
        for word in dom['domain']:
            v = lookup_ft_vector(cursor, word)
            if not (v is None): vecs.append(v)
        if len(vecs) > 0:
            dom['vecs'] = np.array(vecs)
            dom['mean'] = np.mean(np.array(vecs), axis=0)
            yield dom
    db.close()

def reduce_tag_vectors(domains):
    tags = dict()
    for dom in domains:
        tag = dom['tag']
        sum_v = np.sum(dom['vecs'], axis=0)
        n = len(dom['vecs'])
        if not tag in tags:
            tags[tag] = dict(sum=sum_v, n=n)
        else:
            tags[tag]['sum'] += sum_v
            tags[tag]['n'] += n
    for t, v in tags.items():
        v['v'] = v['sum'] / v['n']

    return tags

















