import pandas as pd
import os
import sqlite3
import emb
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
        if df[col].dtype == 'object':
            # cleaning values
            df[col] = df[col].astype(str).str.lower().str.strip()
            all_values = df[col].tolist()
            values = list(set(all_values[:min(segment, len(all_values))]))
            features = emb.get_features(db, values)
            row = ['"' + tablename + '"', inx, '"' + col + '"', '"' + str(df[col].dtype) + '"']
            row.extend(features)
            swriter.writerow(row)
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

