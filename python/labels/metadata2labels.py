import metadata
import json
import os

OPENDATA_LIST = os.environ['OPENDATA_LIST']
LABELS_FILE = os.environ['LABELS_FILE']
LABEL_NAMES_FILE = os.environ['LABEL_NAMES_FILE']
METADATA_DIR = os.environ['METADATA_DIR']
TABLE_LABELS_FILE = os.environ['TABLE_LABELS_FILE']
LABEL_TABLES_FILE = os.environ['LABEL_TABLES_FILE']

labels = {}
if os.path.exists(LABELS_FILE):
    print('loading existing labels')
    labels = json.load(open(LABELS_FILE, 'r'))
table_labels = {}
label_tables = {}
for tablename in open(os.path.join(METADATA_DIR, OPENDATA_LIST), 'r').read().splitlines():
    try:
        print('processing ' + tablename)
        m = metadata.get_metadata(os.path.join(METADATA_DIR, tablename), labels)
        if len(m[0]) > 0:
            labels = m[1]
            table_labels[tablename] = m[0]
    except:
        print('error in reading :%s' % tablename)
        continue
for t, ls in table_labels.items():
    for l in ls:
        if l not in label_tables:
            label_tables[l] = [t]
        else:
            label_tables[l].append(t)

# building the reverse map from label ids to name
label_names = {}
for l, i in labels.items():
    label_names[int(i)] = l
json.dump(label_names, open(LABEL_NAMES_FILE, 'w'))
json.dump(labels, open(LABELS_FILE, 'w'))
json.dump(table_labels, open(TABLE_LABELS_FILE, 'w'))
json.dump(label_tables, open(LABEL_TABLES_FILE, 'w'))
print("number of labels: %d" % len(labels))
print("number of tables with labels: %d" % len(table_labels))
