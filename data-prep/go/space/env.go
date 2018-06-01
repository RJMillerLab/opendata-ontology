package space

import (
	"os"
	"time"
)

// Environment variables
var TableLabelsProbs = os.Getenv("TABLE_LABELS_PROBS")
var QueryResultList = os.Getenv("QUERY_RESULT_LIST")
var LabelsFile = os.Getenv("LABELS_FILE")
var OrgsFile = os.Getenv("ORGS_FILE")
var LabelNamesFile = os.Getenv("LABEL_NAMES_FILE")
var GoodLabelsFile = os.Getenv("GOOD_LABELS_FILE")
var LabelTablesFile = os.Getenv("LABEL_TABLES_FILE")
var FasttextDb = os.Getenv("FASTTEXT_DB")
var DomainEmbsFile = os.Getenv("EMB_SAMPLES_FILE")
var TableEmbsMap = os.Getenv("TABLE_SAMPLE_MAP")
var TransitionProbabilityFile = os.Getenv("TRANSITION_PROB_FILE")
var StateProbabilityFile = os.Getenv("STATE_PROB_FILE")
var TableLabelSelectivityFile = os.Getenv("TABLE_LABEL_SELECTIVITY")

func CheckEnv() {

}

func GetNow() float64 {
	return float64(time.Now().UnixNano()) / 1E9
}
