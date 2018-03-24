import pandas as pd
import os
import sqlite3
#import emb
import sys
import csv

def table2samples(tablename):
    print('processing ' + tablename)
    tablename = tablename.replace('\n', '')
    df = pd.read_json(os.path.join(OUTPUT, 'files', tablename), lines=True)
    # dataframe preparation
    df.columns = [str(col).lower().replace('\s+', '_') for col in df.columns]
    df.replace("\s+", "_", regex=True, inplace=True)
    inx = 0
    for col in df.columns:
        attname = os.path.dirname(tablename) + delimiter + os.path.basename(tablename) + delimiter + str(inx) + delimiter + col + delimiter + str(df[col].dtype)
        if df[col].dtype == 'object':
            # cleaning values
            df[col] = df[col].astype(str).str.lower().str.strip()
        es = df[col].tolist()
        coline = ''
        for i in range(len(es)):
            coline += delimiter + str(es[i])
            if (i+1) % segment == 0 :
                aaf.write(attname + coline + '\n')
                coline = ''
        if len(coline) > 0:
            aaf.write(attname + coline + '\n')
        # domains and counts
        coline = ''
        counts = dict(df[col].value_counts())
        i = 0
        for e, c in counts.items():
            coline += delimiter + str(e) + delimiter + str(c)
            if (i+1) % segment == 0 :
                adf.write(attname + coline + '\n')
                coline = ''
            i += 1
        if len(coline) > 0:
            adf.write(attname + coline + '\n')
        # create emb samples
        #if df[col].dtype == 'object':
        #    features = emb.get_features(db, df[col].tolist())
        #    row = ['"' + tablename + '"', inx, '"' + col + '"', '"' + str(df[col].dtype) + '"']
        #    row.extend(features)
        #    swriter.writerow(row)
        inx += 1
    print('done processing ' + tablename)

    aaf.close()
    adf.close()
    sf.close()


OUTPUT = '/home/fnargesian/FINDOPENDATA_DATASETS/1k'
ALL_ATTS = '/home/fnargesian/FINDOPENDATA_DATASETS/1k/all_attributes'
ALL_DOMS = '/home/fnargesian/FINDOPENDATA_DATASETS/1k/all_domains'
SAMPLES_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/1k/samples'
DB_FILE = '/home/fnargesian/FASTTEXT/fasttext.sqlite3'
#
db = sqlite3.connect(DB_FILE)
#
aaf = open(ALL_ATTS, 'a', newline="\n")
adf = open(ALL_DOMS, 'a', newline="\n")
sf = open(SAMPLES_FILE, 'a', newline="\n")
#
delimiter = ' '
segment = 5000
# header line for emb feature file
fasttextSize = 300
header = ['dataset_name', 'table_name', 'column_id', 'column_name', 'column_type']
for i in range(fasttextSize):
    header.append('f' + str(i))
swriter = csv.writer(sf, delimiter=',', escapechar='\\', lineterminator='\n', quoting=csv.QUOTE_NONE)
swriter.writerow(header)
count = 0
tablename = sys.argv[1]
table2samples(tablename)

