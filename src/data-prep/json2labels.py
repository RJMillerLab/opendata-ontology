import metadata
import json
import os

def get_labels(tablesfile, labelsfile, metadatadir):
    labels = {}
    table_labels = {}
    for tablename in open(os.path.join(metadatadir, tablesfile), 'r').read().splitlines():
        print('processing ' + tablename)
        m = metadata.get_metadata(os.path.join(metadatadir, tablename), labels)
        labels = m[1]
        table_labels[tablename] = m[0]
    json.dump(table_labels, open('result.json', 'w'))
    print(len(labels))
    print(len(table_labels))

OPENDATA_LIST = os.environ['OPENDATA_LIST']
LABELS_FILE = os.environ['LABELS_FILE']
METADATA_DIR = os.environ['OUTPUT_DIR']

get_labels(OPENDATA_LIST, LABELS_FILE, METADATA_DIR)
