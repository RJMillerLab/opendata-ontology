package ontology

import (
	"os"
	"time"
)

// Environment variables
var TableLabelsProbs = os.Getenv("TABLE_LABELS_PROBS")
var QueryResultList = os.Getenv("QUERY_RESULT_LIST")
var LabelsFile = os.Getenv("LABELS_FILE")

func CheckEnv() {

}

func GetNow() float64 {
	return float64(time.Now().UnixNano()) / 1E9
}
