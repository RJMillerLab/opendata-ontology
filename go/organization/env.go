//package ontology
package organization

import (
	"os"
	"time"
)

// Environment variables
var OrgFile = os.Getenv("ORG_FILE")

var LabelNamesFile = os.Getenv("LABEL_NAMES_FILE")
var GoodLabelsFile = os.Getenv("GOOD_LABELS_FILE")
var LabelTablesFile = os.Getenv("LABEL_TABLES_FILE")
var DomainEmbsFile = os.Getenv("EMB_SAMPLES_FILE")
var TableEmbsMap = os.Getenv("TABLE_SAMPLE_MAP")
var FasttextDb = os.Getenv("FASTTEXT_DB")

func CheckEnv() {

}

func GetNow() float64 {
	return float64(time.Now().UnixNano()) / 1E9
}
