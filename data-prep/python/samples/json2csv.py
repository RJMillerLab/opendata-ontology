import pandas as pd
import os
import csv
import sys

def json2csv(line):
    jsonpath = os.path.join(INPUT, line.strip()+".json.gz")
    df = pd.read_json(jsonpath, lines=True, compression="gzip")
    csvpath = os.path.join(OUTPUT, "files", line.strip()+".csv")
    df.to_csv(csvpath, quoting=csv.QUOTE_NONNUMERIC, index=False)
    print(csvpath)

INPUT = "/home/ekzhu/findopendata-datasets-1k"
OUTPUT = "/home/fnargesian/FINDOPENDATA_DATASETS/1k"
#INPUT = os.environ["OPENDATA_DIR"]
#OUTPUT = os.environ["OUTPUT_DIR"]
json2csv(sys.argv[1])
