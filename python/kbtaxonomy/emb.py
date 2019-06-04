import numpy as np
import string
import sqlite3

FT_DB = '/home/fnargesian/FASTTEXT/fasttext.db'

def get_domain_cells(values):
    for val in values:
        val = str(val).strip().strip('"')
        yield [word.lower() for word in val.replace('_', ' ').translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).strip().split(' ')]

def fasttext(c, values):
    features = []
    qmarks = ""
    parts = int(len(values)/100)
    for ix in range(parts):
        qmarks = ",".join("?" for x in range(100))
        sql = "select word, emb from weft where word in (%s)" % qmarks
        c.execute(sql, values[ix*100:(ix+1)*100])
        dt = np.dtype(float)
        dt = dt.newbyteorder('>')
        for row in c.fetchall():
            emb_blob = row[1]
            emb_vec = np.frombuffer(emb_blob, dtype=dt)
            features.append(emb_vec)
        qmarks = ""
    if len(values)%100 != 0:
        qmarks = ",".join("?" for x in range(len(values)%100))
        sql = "select word, emb from fasttext where word in (%s)" % qmarks
        c.execute(sql, values[parts*100:])
        dt = np.dtype(float)
        dt = dt.newbyteorder('>')
        for row in c.fetchall():
            emb_blob = row[1]
            emb_vec = np.frombuffer(emb_blob, dtype=dt)
            features.append(emb_vec)
    return features

def fasttext_small(c, values):
    qmarks = ",".join("?" for x in values)
    sql = "select word, emb from weft where word in (%s)" % qmarks
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
    count = 0
    for f in fasttext(c, cell):
        count += 1
        if feature is None:
            feature = f
        else:
            feature = feature + f
    if feature is not None:
        feature = feature/count
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

def get_vals_fasttext(values):
    fcs = []
    db = sqlite3.connect(FT_DB)
    cursor = db.cursor()
    for val in values:
        ft_val = fasttext_cells(cursor, get_domain_cells([val]))
        if len(ft_val) > 0:
            fcs.append(val)
    cursor.close()
    return fcs

def get_features(values):
    db = sqlite3.connect(FT_DB)
    cursor = db.cursor()
    ft_cells = fasttext_cells(cursor, get_domain_cells(values))
    if len(ft_cells) > 0:
        return np.average(ft_cells, axis=0).tolist()
    cursor.close()
    return np.array([])
