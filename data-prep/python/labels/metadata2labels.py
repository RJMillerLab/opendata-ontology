import metadata
import json
import os

def get_labels(tablesfile, labelsfile, tablelabelsfile, metadatadir):
    labels = {}
    table_labels = {}
    for tablename in open(os.path.join(metadatadir, tablesfile), 'r').read().splitlines():
        print('processing ' + tablename)
        m = metadata.get_metadata(os.path.join(metadatadir, tablename), labels)
        if len(m[0]) > 0:
            labels = m[1]
            table_labels[tablename] = m[0]
    json.dump(labels, open(labelsfile, 'w'))
    json.dump(table_labels, open(tablelabelsfile, 'w'))
    print(len(labels))
    print(len(table_labels))

OPENDATA_LIST = os.environ['OPENDATA_LIST']
LABELS_FILE = os.environ['LABELS_FILE']
METADATA_DIR = os.environ['METADATA_DIR']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
get_labels(OPENDATA_LIST, LABELS_FILE, TABLE_LABELS_FILE, METADATA_DIR)
