OPENDATA_DIR = /home/fnargesian/FINDOPENDATA_DATASETS/1k
OUTPUT_DIR = /home/fnargesian/FINDOPENDATA_DATASETS/1k
OPENDATA_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/1k/datasets-1k.list
#OPENDATA_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/1k/datasets-1k.debug
METADATA_DIR = /home/ekzhu/findopendata-datasets-1k
EMB_SAMPLES_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/1k/emb.samples
LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/1k/labels.json
LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/1k/small_labels.json
TABLE_LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/1k/table_labels.json
FT_FILE = /home/fnargesian/FASTTEXT/fasttext.sqlite3
GOOD_LABELS_FILE = /home/fnargesian/FINDOPENDATA_DATASETS/1k/good_labels.json
EMB_FILES_LIST = /home/fnargesian/FINDOPENDATA_DATASETS/1k/emb.files
TABLE_SAMPLE_MAP = /home/fnargesian/FINDOPENDATA_DATASETS/1k/table.samples
LABEL_MODEL_SOFTMAX = /home/fnargesian/FINDOPENDATA_DATASETS/1k/model.softmax
LABEL_MODEL_SOFTPROB = /home/fnargesian/FINDOPENDATA_DATASETS/1k/model.softprob

build:
	go build ./...

install:
	go install ./...

step0:
	OPENDATA_LIST=$(OPENDATA_LIST) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	OPENDATA_DIR=$(OPENDATA_DIR) \
	cat $(OPENDATA_LIST) | xargs -P 30 -n 1 python python/json2csv.py

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
	python python/labels/find_good_labels.py

step7: embfiles
	EMB_FILES_LIST=$(EMB_FILES_LIST) \
	EMB_SAMPLES_FILE=$(EMB_SAMPLES_FILE) \
	GOOD_LABELS_FILE=$(GOOD_LABELS_FILE) \
	TABLE_LABELS_FILE=$(TABLE_LABELS_FILE) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	TABLE_SAMPLE_MAP=$(TABLE_SAMPLE_MAP) \
	python python/samples/emb2sample.py

step8: 
	LABEL_MODEL_SOFTMAX=$(LABEL_MODEL_SOFTMAX) \
	LABEL_MODEL_SOFTPROB=$(LABEL_MODEL_SOFTPROB) \
	EMB_SAMPLES_FILE=$(EMB_SAMPLES_FILE) \
	python python/labels/learn_labels.py

embfiles:
	cd $(OUTPUT_DIR)/domains; find -name "*.ft-mean" > $(EMB_FILES_LIST) 
