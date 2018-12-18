import json

gls = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/good_labels_20k.json', 'r'))
labels = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_names_20k.json', 'r'))
label_tables = json.load(open('/home/fnargesian/FINDOPENDATA_DATASETS/10k/label_tables_20k.json', 'r'))

label_index = {}
label_overlap = {}
label_size = {}
for l1, l1ts in label_tables.items():
    if int(l1) in gls:
        label_size[l1] = len(l1ts)
        label_index[l1] = []
        label_overlap[l1] = {}
        for l2, l2ts in label_tables.items():
            if int(l2) in gls:
                i = set(l1ts).intersection(set(l2ts))
                if len(i) != 0:
                    if l1 not in label_index:
                        label_index[l1] = []
                        label_overlap[l1] = {}
                    label_index[l1].append(l2)
                    label_overlap[l1][l2] = len(i)
for l, ls in label_index.items():
    print("l %s %d" % (l, len(ls)))
