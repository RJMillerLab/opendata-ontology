package ontology

import (
	"os"
	"time"
)

// Environment variables
var TableLabelsProbs = os.Getenv("TABLE_LABELS_PROBS")

func CheckEnv() {

}

func GetNow() float64 {
	return float64(time.Now().UnixNano()) / 1E9
}
