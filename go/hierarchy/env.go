package hierarchy

import (
	"os"
	"time"
)

// Environment variables
var TableTagsProbs = os.Getenv("TABLE_LABELS_PROBS")
var QueryResultList = os.Getenv("QUERY_RESULT_LIST")
var TagsFile = os.Getenv("LABELS_FILE")
var OrgsFile = os.Getenv("ORGS_FILE")
var TagNamesFile = os.Getenv("LABEL_NAMES_FILE")
var GoodTagsFile = os.Getenv("GOOD_LABELS_FILE")
var TagTablesFile = os.Getenv("LABEL_TABLES_FILE")
var FasttextDb = os.Getenv("FASTTEXT_DB")
var DomainEmbsFile = os.Getenv("EMB_SAMPLES_FILE")
var TableEmbsMap = os.Getenv("TABLE_SAMPLE_MAP")
var TransitionProbabilityFile = os.Getenv("TRANSITION_PROB_FILE")
var StateProbabilityFile = os.Getenv("STATE_PROB_FILE")
var TableTagSelectivityFile = os.Getenv("TABLE_LABEL_SELECTIVITY")
var TablesFile = os.Getenv("TABLES_FILE")
var TablesDir = os.Getenv("TABLES_DIR")

func CheckEnv() {

}

func GetNow() float64 {
	return float64(time.Now().UnixNano()) / 1E9
}
