import pandas as pd
import io
import gzip
import math
import os
import csv
import sys

def json2csv(line):
    try:
        print(line)
        jsonpath = os.path.join(INPUT, line.strip()+".json.gz")
        with gzip.open(jsonpath, 'rb') as f:
            size = f.seek(0, io.SEEK_END)
            if size > 3*math.pow(10,9):
                print('%s is too large' % jsonpath)
                return
        df = pd.read_json(jsonpath, lines=True, compression="gzip")
        csvpath = os.path.join(OUTPUT, "files", line.strip()+".csv")
        os.makedirs(os.path.dirname(csvpath), exist_ok=True)
        df.to_csv(csvpath, quoting=csv.QUOTE_NONNUMERIC, index=False)
        print('saved %s' % csvpath)
    except:
        print('exception raised while reading %s' % line.strip())
        raise

INPUT = "/home/ekzhu/findopendata-datasets"
OUTPUT = "/home/fnargesian/FINDOPENDATA_DATASETS/10k"
#INPUT = os.environ["OPENDATA_DIR"]
#OUTPUT = os.environ["OUTPUT_DIR"]
json2csv(sys.argv[1])
