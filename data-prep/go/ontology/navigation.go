package ontology

import (
	"log"
	"strconv"
	"strings"
	"unicode"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/embedding"
)

var (
	labels      []string
	labelEmbs   map[string][]float64
	labelTables map[string][]string
)

type navigation struct {
	paths [][]state
}

type state struct {
	labels   []string
	semantic [][]float64
	tables   []string
}

func InitializeNavigationSimulation() {
	// load labels
	labelIds := make([]int, 0)
	labelTables = make(map[string][]string)
	err := loadJson(GoodLabelsFile, &labelIds)
	if err != nil {
		panic(err)
	}
	labelIds = labelIds[:20]
	labelNames := make(map[string]string)
	err = loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	labels = make([]string, 0)
	for _, gl := range labelIds {
		labels = append(labels, labelNames[strconv.Itoa(gl)])
	}
	log.Printf("lables: %v", labels)
	// load label table
	err = loadJson(LabelTablesFile, &labelTables)
	if err != nil {
		panic(err)
	}
	// load the embedding of each label
	getLabelsEmbedding()
	//nav := navigation{paths=make([][]state, 0),}
}

func getLabelsEmbedding() {
	ft, err := InitInMemoryFastText(FasttextDb, func(v string) []string {
		stopWords := []string{"ckan_topiccategory_", "ckan_keywords_", "ckan_tags_", "ckan_subject_", "socrata_domaincategory_", "socrata_domaintags_", "socrata_tags_"}
		for _, st := range stopWords {
			v = strings.Replace(v, st, "", -1)
		}
		v = strings.Replace(strings.Replace(v, "_", " ", -1), "-", " ", -1)
		return strings.Split(v, " ")
	}, func(v string) string {
		return strings.ToLower(strings.TrimFunc(strings.TrimSpace(v), unicode.IsPunct))
	})
	if err != nil {
		panic(err)
	}
	labelEmbs := make(map[string][]float64)
	for _, label := range labels {
		embVec, err := ft.GetPhraseEmbMean(label)
		if err != nil {
			log.Printf("Error in building embedding for label %s : %s\n", label, err.Error())
			// TODO: what to do with labels with no embedding
			continue
		}
		labelEmbs[label] = embVec
	}
}

func getSemanticCoherence(st state) float64 {
	return 0.0
}

// evaluate the semantic relevance of states (their label sets) by
// computing cosine on their aggregate embedding vectors
func SemanticRelevance(state1, labels2 state) float64 {
	return 0.0
}

// evaluate the is-A quality of two label by computing the co-occurrence of
// labels for tables
func IsAScore(state1, state2 state) float64 {
	return 0.0
}

// given a navigation path (consisting of ordered list of label sets), generate next
// possible states. Make sure the labels are not revisited and the semantic relevance is above a threshold.
func GenerateNextStates(path [][][]float64) {
}

// simulate user navigation
func Simulate() {
	// pick a start state

	// generate next states

	// randomly pick the next state

	// check termination condition: the next state list is empty or all tables are reachable
}
