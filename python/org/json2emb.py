import pandas as pd
import os
import sqlite3
import emb
import sys
import csv


def table2embs(tablename):
    print('processing ' + tablename)
    tablename = tablename.replace('\n', '')
    try:
        #df = pd.read_json(os.path.join(OUTPUT, 'files', tablename), lines=True)
        df = pd.read_csv(os.path.join(OUTPUT, 'files', tablename))#, lines=True)
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
                if len(features) > 0:
                    row = ['"' + tablename + '"', inx, '"' + col + '"', str(df[col].dtype)]
                    row.extend([str(f) for f in features])
                    swriter.writerow(row)
            inx += 1
    except:
        print('error: pass')
        pass

    print('done processing ' + tablename)

    #sf.close()


def write_header():
    sf = open(EMBS_FILE, 'a', newline="\n")
    header = ['dataset_name', 'column_id', 'column_name', 'column_type']
    for i in range(fasttextSize):
        header.append('f' + str(i))
    swriter = csv.writer(sf, delimiter=',', escapechar='\\', lineterminator='\n', quoting=csv.QUOTE_NONE)
    swriter.writerow(header)
    sf.close()


OUTPUT = '/home/fnargesian/FINDOPENDATA_DATASETS/10k'
EMBS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_embs'
DB_FILE = '/home/kenpu/clones/tagcloud-nlp-generator/ft.sqlite3'
#
db = sqlite3.connect(DB_FILE)
#
sf = open(EMBS_FILE, 'a', newline="\n")
#
segment = 1000
# header line for emb feature file
fasttextSize = 300
#header = ['dataset_name', 'table_name', 'column_id', 'column_name', 'column_type']
#for i in range(fasttextSize):
#    header.append('f' + str(i))
swriter = csv.writer(sf, delimiter=',', escapechar='\\', lineterminator='\n', quoting=csv.QUOTE_NONE)
#swriter.writerow(header)
tablename = sys.argv[1]
table2embs(tablename)

