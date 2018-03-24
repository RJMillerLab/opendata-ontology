import sqlite3
import numpy as np
import re

def get_domain_cells(values):
    for val in values:
        val = val.strip().strip('"')
        yield [word.lower() for word in re.findall(r'\w+', val)]

def get_union_cells(dom_file1, dom_file2):
    u1 = list(get_domain_cells(dom_file1)) + list(get_domain_cells(dom_file2))
    u2 = set(tuple(x) for x in u1)
    union = [list(x) for x in u2]
    return union

def fasttext(c, values):
    qmarks = ",".join("?" for x in values)
    sql = "select word, emb from fasttext where word in (%s)" % qmarks
    c.execute(sql, values)
    dt = np.dtype(float)
    dt = dt.newbyteorder('>')
    features = []
    for row in c.fetchall():
        emb_blob = row[1]
        emb_vec = np.frombuffer(emb_blob, dtype=dt)
        features.append(emb_vec)
    return features

def fasttext_cell(c, cell):
    feature = None
    for f in fasttext(c, cell):
        if feature is None:
            feature = f
        else:
            feature = feature + f
    return feature

def fasttext_cells(c, cells):
    features = []
    for cell in cells:
        f = fasttext_cell(c, cell)
        if f is not None:
            features.append(f)
    return np.array(features)

def average(features):
    return np.array([np.average(features, axis=0)])

def sum(features):
    s = np.sum(features, axis=0)
    return np.array([s])

def get_features(db, values):
    #db = sqlite3.connect("/home/fnargesian/FASTTEXT/fasttext.sqlite3")
    cursor = db.cursor()
    ft_cells = fasttext_cells(cursor, get_domain_cells(values))
    if len(ft_cells) > 0:
        return np.average(ft_cells, axis=0).tolist()
    else:
        return []
