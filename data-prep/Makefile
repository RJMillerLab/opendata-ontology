OPENDATA_DIR = /home/fnargesian/FINDOPENDATA_DATASETS/10k
#OPENDATA_DIR = /home/ekzhu/findopendata-datasets
OUTPUT_DIR = /home/fnargesian/FINDOPENDATA_DATASETS/10k
OPENDATA_JSON_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-10k.list
OPENDATA_JSON_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-20k.list
OPENDATA_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-10k.list
OPENDATA_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/datasets-20k.list
METADATA_DIR = /home/ekzhu/findopendata-datasets
EMB_SAMPLES_CSV_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/samples/emb_samples.csv
EMB_SAMPLES_CSV_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/samples/emb_samples_20k.csv
LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/labels.json
LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/labels_20k.json
TABLE_LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels.json
TABLE_LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/table_labels_20k.json
FT_FILE = /home/fnargesian/FASTTEXT/fasttext.sqlite3
GOOD_LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/good_labels.json
GOOD_LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/good_labels_20k.json
EMB_FILES_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/emb.files
EMB_FILES_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/emb_20k.files
TABLE_SAMPLE_MAP = /home/fnargesian/FINDOPENDATA_DATASETS/10k/table.samples
TABLE_SAMPLE_MAP = /home/fnargesian/FINDOPENDATA_DATASETS/10k/table_20k.samples
NUM_LABELS = 100
NUM_TABLE_SAMPLES = 100
EMB_PARAM_FILE = /home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/labels/params_5k_3.json
EMB_PARAM_FILE = /home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/labels/params_10k_2.json
EMB_PARAM_FILE = /home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/labels/params_10k_100.json
NUM_PARAM_FILE = /home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/labels/params_5k_3.json
NUM_FILES_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/10k/minhash.files
NUM_SAMPLES_CSV_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/num_samples.csv
LABEL_EMB_CSAMPLE_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/emb_csample_files.json
LABEL_EMB_CSAMPLE_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/emb_csample_files_20k.json
LABEL_EMB_DSAMPLE_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/emb_dsample_files.json
LABEL_EMB_DSAMPLE_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/emb_dsample_files_20k.json
ALL_EMB_SAMPLE_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/10k/samples/emb.all
MODEL_DIR = /home/fnargesian/FINDOPENDATA_DATASETS/10k/models

build:
	go build ./...

install:
	go install ./...

step0:
	OPENDATA_JSON_LIST=$(OPENDATA_JSON_LIST) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_DIR=$(OPENDATA_DIR) \
	cat $(OPENDATA_JSON_LIST) | xargs -d '\n' -P 20 -n 1 python python/samples/json2csv.py

step1:
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_LIST=$(OPENDATA_LIST) \
	python python/setup.py

step2:
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_LIST=$(OPENDATA_LIST) \
	OPENDATA_DIR=$(OPENDATA_DIR) \
	go run go/cmd/build_domain_values/main.go

step3:
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_LIST=$(OPENDATA_LIST) \
	OPENDATA_DIR=$(OPENDATA_DIR) \
	go run go/cmd/classify_domain_values/main.go

step4:
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_LIST=$(OPENDATA_LIST) \
	OPENDATA_DIR=$(OPENDATA_DIR) \
	go run go/cmd/build_domain_embs/main.go

step5:
	OPENDATA_LIST=$(OPENDATA_LIST) \
	METADATA_DIR=$(METADATA_DIR) \
	LABELS_FILE=$(LABELS_FILE) \
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	python python/labels/metadata2labels.py

step6:
	GOOD_LABELS_FILE=$(GOOD_LABELS_FILE) \
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	NUM_LABELS=$(NUM_LABELS) \
	LABELS_FILE=$(LABELS_FILE) \
	python python/labels/find_good_labels.py

step7: embfiles
	EMB_FILES_LIST=$(EMB_FILES_LIST) \
	EMB_SAMPLES_CSV_FILE=$(EMB_SAMPLES_CSV_FILE) \
	GOOD_LABELS_FILE=$(GOOD_LABELS_FILE) \
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	TABLE_SAMPLE_MAP=$(TABLE_SAMPLE_MAP) \
	NUM_TABLE_SAMPLES=$(NUM_TABLE_SAMPLES) \
	python python/samples/emb2sample.py

step8: 
	EMB_SAMPLES_CSV_FILE=$(EMB_SAMPLES_CSV_FILE) \
	EMB_PARAM_FILE=$(EMB_PARAM_FILE) \
	python python/labels/model_labels.py

step9:
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_LIST=$(OPENDATA_LIST) \
	OPENDATA_DIR=$(OPENDATA_DIR) \
	go run go/cmd/build_domain_minhash/main.go

step10: minhashfiles
	NUM_FILES_LIST=$(NUM_FILES_LIST) \
	NUM_SAMPLES_CSV_FILE=$(NUM_SAMPLES_CSV_FILE) \
	GOOD_LABELS_FILE=$(GOOD_LABELS_FILE) \
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	TABLE_SAMPLE_MAP=$(TABLE_SAMPLE_MAP) \
	NUM_TABLE_SAMPLES=$(NUM_TABLE_SAMPLES) \
	python python/samples/minhash2sample.py

step11: 
	NUM_SAMPLES_CSV_FILE=$(NUM_SAMPLES_CSV_FILE) \
	NUM_PARAM_FILE=$(NUM_PARAM_FILE) \
	python python/labels/num_model_binary_labels.py
	#python python/labels/num_model_labels.py

step12: embfiles
	EMB_FILES_LIST=$(EMB_FILES_LIST) \
	EMB_SAMPLES_CSV_FILE=$(EMB_SAMPLES_CSV_FILE) \
	GOOD_LABELS_FILE=$(GOOD_LABELS_FILE) \
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	TABLE_SAMPLE_MAP=$(TABLE_SAMPLE_MAP) \
	NUM_TABLE_SAMPLES=$(NUM_TABLE_SAMPLES) \
	LABEL_EMB_CSAMPLE_FILE=$(LABEL_EMB_CSAMPLE_FILE) \
	ALL_EMB_SAMPLE_FILE=$(ALL_EMB_SAMPLE_FILE) \
	python python/samples/sep_model_emb2sample.py

step13: 
	ALL_EMB_SAMPLE_FILE=$(ALL_EMB_SAMPLE_FILE) \
	EMB_PARAM_FILE=$(EMB_PARAM_FILE) \
	LABEL_EMB_CSAMPLE_FILE=$(LABEL_EMB_CSAMPLE_FILE) \
	LABEL_EMB_MODEL_FILE=$(LABEL_EMB_MODEL_FILE) \
	MODEL_DIR=$(MODEL_DIR) \
	python python/labels/model_all_labels.py

step14: 
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	LABEL_EMB_MODEL_FILE=$(LABEL_EMB_MODEL_FILE) \
	TABLE_BOOSTED_LABELS_FILE=$(TABLE_BOOSTED_LABELS_FILE) \
	TABLE_SAMPLE_MAP=$(TABLE_SAMPLE_MAP) \
	ALL_EMB_SAMPLE_FILE=$(ALL_EMB_SAMPLE_FILE) \
	python python/annotate/annotate_table.py


embfiles:
	cd $(OUTPUT_DIR)/domains; find . -name "*.ft-mean" > $(EMB_FILES_LIST) 

minhashfiles:
	cd $(OUTPUT_DIR)/domains; find . -name "*.minhash" > $(NUM_FILES_LIST) 
