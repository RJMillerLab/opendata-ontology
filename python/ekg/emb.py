import numpy as np
from scipy.spatial import distance
import re

def get_domain_cells(values):
    ws = []
    for val in values:
        val = str(val).strip().strip('"')
        ws.extend([word.lower() for word in re.findall(r'\w+', val)])
    return ws

def fasttext(c, values):
    features = []
    qmarks = ""
    qmarks = ",".join("?" for x in values)
    sql = "select word, emb from fasttext where word in (%s)" % qmarks
    c.execute(sql, values)
    dt = np.dtype(float)
    dt = dt.newbyteorder('>')
    for row in c.fetchall():
        emb_blob = row[1]
        emb_vec = np.frombuffer(emb_blob, dtype=dt)
        features.append(emb_vec)
    return features

def semantic(db, values):
    c = db.cursor()
    return fasttext(c, get_domain_cells(values))

def semantic_group_sim(fs1, fs2):
    v1 = np.array([np.average(fs1, axis=0)])
    v2 = np.array([np.average(fs2, axis=0)])
    return 1.0 - distance.cosine(v1, v2)

def semantic_group_sim2(fs1, fs2):
    sim = 0.0
    for f1 in fs1:
        for f2 in fs2:
            sim += 1.0 - distance.cosine(f1, f2)
    return sim/float(len(fs1)*len(fs2))

