import os
import json

INPUT_DIR = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/files'
TABLE_LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/table_labels_500.json'
LABEL_TABLES_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/label_tables_500.json'
LABELS_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/labels_500.json'
LABEL_NAMES_FILE = '/home/fnargesian/FINDOPENDATA_DATASETS/synthetic/label_names_500.json'

domain_tags = dict()
tag_domains = dict()
tag_names = dict()
tags = []
tag_ids = dict()

for f in os.listdir(INPUT_DIR):
    if f.endswith('tags.vec'):
        line = open(os.path.join(INPUT_DIR, f)).readlines()[0]
        domain = f.replace('tags.vec', 'csv')
        tag = line.split(' ')[0]
        if tag not in tag_ids:
            tag_ids[tag] = len(tag_ids)
            tags.append(tag_ids[tag])
            tag_names[str(tag_ids[tag])] = tag
        if tag_ids[tag] not in tag_domains:
            tag_domains[tag_ids[tag]] = []
        tag_domains[tag_ids[tag]].append(domain)
        if domain not in domain_tags:
            domain_tags[domain] = []
        domain_tags[domain].append(tag_ids[tag])
print(len(tag_domains))
print(len(domain_tags))
print(len(tag_names))
print(len(tags))
json.dump(tag_domains, open(LABEL_TABLES_FILE, 'w'))
json.dump(domain_tags, open(TABLE_LABELS_FILE, 'w'))
json.dump(tag_names, open(LABEL_NAMES_FILE, 'w'))
json.dump(tags, open(LABELS_FILE, 'w'))



