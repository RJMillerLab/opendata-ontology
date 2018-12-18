import json
import csv

TABLE_BOOSTED_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/tables_31k.boosted_labels'
TABLE_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_31k.json'
LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/labels_all.json'
LABEL_NAMES_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.json'
LABEL_TABLES_CSV_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_tables_31k.csv'
LABEL_NAMES_CSV_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_31k.csv'

def print_boosted_labels_stats():
    tableBoostedLabels = json.load(open(TABLE_BOOSTED_LABELS_FILE, 'r'))
    tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
    labelNames = json.load(open(LABEL_NAMES_FILE, 'r'))
    tableBLs = {}
    tableLs = {}
    for k, v in tableLabels.items():
        tableLs[k] = len(v)
    for t, ls in tableBoostedLabels.items():
        bls = {}
        for l, p in ls.items():
            if p != 0.0 and p != 1.0:
                print("t: %s, l: %s, p: %f" % (k, l, p))
            if int(l) not in tableLabels[t]:
                bls[l] = labelNames[l]
        if len(bls) > 0:
            tableBLs[t] = len(bls)
            #print(t)
            #print(bls)
            #print("-----------------")
    print("number of tables with labels: %d" % len(tableLabels))
    print("avg number of labels: %f" % (sum(tableLs.values())/float(len(tableLs))))
    print("number of tables with boosted labels: %d" % len(tableBLs))
    print("avg number of boosted labels: %f" % (sum(tableBLs.values())/float(len(tableBLs))))


def persist_table_labels():
    tableLabels = json.load(open(TABLE_LABELS_FILE, 'r'))
    tableBoostedLabels = json.load(open(TABLE_BOOSTED_LABELS_FILE, 'r'))
    csf = open(LABEL_TABLES_CSV_FILE, 'w')
    cswriter = csv.writer(csf, delimiter='|', lineterminator='\n', quoting=csv.QUOTE_NONE)
    cswriter.writerow(['label_id', 'table_name', 'prob', 'source'])
    for t, ls in tableBoostedLabels.items():
        for l, p in ls.items():
            row = []
            if int(l) not in tableLabels[t]:
                row = [str(l), str(t), p, "b"]
            else:
                row = [str(l), str(t), p, "m"]
            cswriter.writerow(row)
    csf.close()

def persist_label_names():
    labelNames = json.load(open(LABEL_NAMES_FILE, 'r'))
    csf = open(LABEL_NAMES_CSV_FILE, 'w')
    cswriter = csv.writer(csf, delimiter='|', lineterminator='\n', quoting=csv.QUOTE_NONE)
    cswriter.writerow(['id', 'name'])
    for k,v in labelNames.items():
        v = v.replace("\"", "")
        #print("%s %s" % (k,v))
        cswriter.writerow([k,v])
    csf.close()

#print_boosted_labels_stats()
persist_label_names()
persist_table_labels()
