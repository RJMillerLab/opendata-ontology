//package ontology
package organization

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
	"unicode"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/embedding"
)

type state struct {
	labels    map[string]bool
	sem       []float64
	tables    map[string]bool
	coherence float64
	name      string
}

func InitializeNavigation() {
	// load labels
	labelIds := make([]int, 0)
	lts := make(map[string][]string)
	labelTables = make(map[string][]string)
	labels = make(map[string]bool)
	labelEmbs = make(map[string][]float64)
	labelsList = make([]string, 0)
	err := loadJson(GoodLabelsFile, &labelIds)
	if err != nil {
		panic(err)
	}
	labelIds = labelIds //[:20]
	labelNames := make(map[string]string)
	err = loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	// load label table
	err = loadJson(LabelTablesFile, &lts)
	if err != nil {
		panic(err)
	}
	for _, gl := range labelIds {
		labels[labelNames[strconv.Itoa(gl)]] = true
		//labelsList = append(labelsList, labelNames[strconv.Itoa(gl)])
		//labelTables[labelNames[strconv.Itoa(gl)]] = lts[strconv.Itoa(gl)]
	}
	// load the embedding of each label
	getLabelEmbeddings()
	// eliminate labels without semantics
	for _, gl := range labelIds {
		if _, ok := labelEmbs[labelNames[strconv.Itoa(gl)]]; ok {
			labelsList = append(labelsList, labelNames[strconv.Itoa(gl)])
			labelTables[labelNames[strconv.Itoa(gl)]] = lts[strconv.Itoa(gl)]
		} else {

		}
	}
}

func getLabelEmbeddings() {
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
	for label, _ := range labels {
		embVec, err := ft.GetPhraseEmbMean(label)
		if err != nil {
			fmt.Printf("Error in building embedding for label %s : %s\n", label, err.Error())
			// TODO:
			// what
			// to
			// do
			// with
			// labels
			// with
			// no
			// embedding
			continue
		}
		labelEmbs[label] = embVec
	}
	return
}

func getSemanticCoherence(st state) float64 {
	return 0.0
}

// evaluate
// the
// semantic
// relevance
// of
// states
// (their
// label
// sets)
// by
// computing
// cosine
// on
// their
// aggregate
// embedding
// vectors
func getSemanticRelevance(state1, state2 state) float64 {
	return Cosine(state1.sem, state2.sem)
}

func getStatesCoverage(state1, state2 state) float64 {
	overlap := 0
	for t, _ := range state1.tables {
		if _, ok := state2.tables[t]; ok {
			overlap += 1
		}
	}
	return float64(overlap) / float64(len(state1.tables))
	//return float64(len(state1.tables)+len(state2.tables)-overlap) / float64(len(state1.tables))
}

// evaluate
// the
// is-A
// quality
// of
// two
// label
// by
// computing
// the
// co-occurrence
// of
// labels
// for
// tables
func isAScore(state1, state2 state) float64 {
	relevance := getSemanticRelevance(state1, state2)
	//coverage
	//:=
	//getStatesCoverage(state1,
	//state2)
	//return
	//relevance
	//*
	//coverage
	return relevance
	//return
	//coverage
}

func generateStartState() state {
	li := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(labelsList))
	label := labelsList[li]
	st := state{
		labels: map[string]bool{label: true},
		sem:    labelEmbs[label],
		tables: SliceToMap(labelTables[label]),
		name:   label,
	}
	return st
}

// given
// a
// navigation
// path
// (consisting
// of
// ordered
// list
// of
// label
// sets),
// generate
// next
// possible
// states.
// Make
// sure
// the
// labels
// are
// not
// revisited
// and
// the
// semantic
// relevance
// is
// above
// a
// threshold.
func (navPath *path) generateNextState() bool {
	nextState := state{}
	// enumerate
	// possible
	// states
	allNextStates := enumerateAllNextStates(navPath)
	// should
	// the
	// path
	// be
	// terminated?
	if len(allNextStates) == 0 {
		log.Println("no next state")
		return false
	}
	// generate
	// the
	// prob
	// distribution
	// on
	// possible
	// states
	// compute
	// semantic
	// relevance
	// of
	// the
	// current
	// state
	// to
	// the
	// potential
	// next
	// state:
	// cosine
	// of
	// embs
	scores, _, cdf := generateNextStateDistribution(allNextStates, navPath.states[len(navPath.states)-1])
	// should
	// the
	// path
	// be
	// terminated?
	maxScore := Max(scores)
	if maxScore < relevanceThreshold {
		log.Printf("terminating because sem rel is below the threshold: %f\n", maxScore)
		return false
	}
	// choose
	// a
	// state
	// based
	// on
	// the
	// distribution
	nextStateNotFound := true
	//checkedStateIds
	//:=
	//make(map[int]bool)
	attemptCount := 0
	for nextStateNotFound && attemptCount < 10 { //len(checkedStateIds) < len(allNextStates) && len(checkedStateIds) < 10 {
		attemptCount += 1
		newsid := pickAState(cdf)
		nextState = allNextStates[newsid]
		nextStateNotFound = !navPath.updatePath(nextState, scores[newsid])
	}
	return !nextStateNotFound //, navPath
}

func pickAState(cdf []float64) int {
	r := rand.New(rand.NewSource(time.Now().UnixNano())).Float64()
	for i, f := range cdf {
		if r < f {
			return i
		}
	}
	return -1
}

func pickStates(cdf []float64) []int {
	is := make([]int, 0)
	for float64(len(is)) < math.Max(10.0, float64(len(cdf))) {
		r := rand.New(rand.NewSource(time.Now().UnixNano())).Float64()
		for i, f := range cdf {
			if r < f {
				is = append(is, i)
			}
		}
	}
	return is
}

// this
// method
// returns
// the
// CDF
// of
// the
// next
// state
// distribution
func generateNextStateDistribution(states []state, prevState state) ([]float64, []float64, []float64) {
	rels := make([]float64, 0)
	ps := make([]float64, 0)
	cdf := make([]float64, 0)
	sum := 0.0
	for _, st := range states {
		//r
		//:=
		//getSemanticRelevance(prevState,
		//st)
		// discard
		// states
		// with
		// no
		// embedding
		if len(st.sem) == 0 {
			continue
		}
		r := isAScore(st, prevState)
		rels = append(rels, r)
		sum += r
	}
	acc := 0.0
	for _, r := range rels {
		p := r / sum
		ps = append(ps, p)
		cdf = append(cdf, p+acc)
		acc += p
	}
	return rels, ps, cdf
}

func enumerateAllNextStates(navPath *path) []state {
	states := make([]state, 0)
	for label, _ := range navPath.unseenLabels {
		sem := labelEmbs[label]
		st := state{
			labels:    map[string]bool{label: true},
			sem:       sem,
			tables:    SliceToMap(labelTables[label]),
			coherence: 0.0,
		}
		states = append(states, st)
	}
	return states
}

func (s *state) endsPath(p path) bool {
	cts := make(map[string]bool)
	for t, _ := range p.coveredTables {
		if _, ok := s.tables[t]; ok {
			cts[t] = true
		}
	}
	if len(cts) == 0 {
		return true
	}
	return false
}

func (p *path) updatePath(st state, prevStateRelevance float64) bool {
	// find the overlap of the tables covered by the path and those covered by the new state
	cts := make(map[string]bool)
	for t, _ := range p.coveredTables {
		if _, ok := st.tables[t]; ok {
			cts[t] = true
		}
	}
	if len(cts) == 0 {
		return false
	}
	p.coveredTables = cts
	p.states = append(p.states, st)
	for l, _ := range st.labels {
		p.seenLabels[l] = true
		delete(p.unseenLabels, l)
	}
	p.isascores = append(p.isascores, prevStateRelevance)
	return true
}

// simulate
// user
// navigation
func Simulate() {
	for i := 0; i <= 20; i = len(seenPaths) {
		// pick
		// a
		// start
		// state
		start := generateStartState()
		navPath := newPath(start)
		termination := false
		for ok := false; !ok; ok = termination {
			// generate
			// next
			// states
			// and
			// update
			// path
			// with
			// the
			// new
			// state
			openend := navPath.generateNextState()
			if openend == false {
				termination = true
			} else if len(navPath.coveredTables) <= 1 {
				fmt.Print("terminating because we reached a single table")
				termination = true
			}
		}
		if len(navPath.states) > 1 && navPath.seenPath() == false {
			fmt.Printf("simulation %d\n", i)
			printPath(navPath)
			fmt.Println("----------------------------------")
			seenPaths = append(seenPaths, navPath)
		}
	}
}

func printPath(p path) {
	for i, s := range p.states {
		ls := make([]string, 0)
		for k, _ := range s.labels {
			ls = append(ls, k)
		}
		fmt.Printf("state %d : %v (tables: %d) covered tables %d\n", i, ls, len(s.tables), len(p.coveredTables))
	}
}

func (p *path) seenPath() bool {
	for _, op := range seenPaths {
		if p.equalPath(op) == true {
			return true
		}
	}
	return false
}
