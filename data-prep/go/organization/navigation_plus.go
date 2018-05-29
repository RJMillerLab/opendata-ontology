package ontology

import (
	"fmt"
	"log"
	"strconv"
)

var (
	states             []string
	labels             map[string]bool
	tables             []string
	labelNames         map[string]string
	labelsList         []string
	labelEmbs          map[string][]float64
	labelDomainEmbs    map[string][][]float64
	labelAvgEmb        map[string][]float64
	tableEmbsMap       map[string][]int
	labelTables        map[string][]string
	domainEmbs         [][]float64
	seenPaths          []path
	relevanceThreshold = 0.1
	z                  = 10.0
)

type path struct {
	tablename       string
	states          []state
	isascores       []float64
	seenLabels      map[string]bool
	unseenLabels    map[string]bool
	coveredTables   map[string]bool
	probability     float64
	transitionProbs []float64
}

func newPath(startState state) path {
	ls := CopyMap(labels)
	for l, _ := range startState.labels {
		delete(ls, l)
	}
	p := path{
		states:          []state{startState},
		seenLabels:      CopyMap(startState.labels),
		unseenLabels:    ls,
		isascores:       []float64{0.0},
		coveredTables:   CopyMap(startState.tables),
		transitionProbs: make([]float64, 0),
	}
	return p
}

func InitializeNavigationPlus() {
	// load labels
	labelIds := make([]int, 0)
	lts := make(map[int][]string)
	labelTables = make(map[string][]string)
	labels = make(map[string]bool)
	labelEmbs = make(map[string][]float64)
	labelsList = make([]string, 0)
	err := loadJson(GoodLabelsFile, &labelIds)
	if err != nil {
		panic(err)
	}
	labelIds = labelIds //[:20]
	labelNames = make(map[string]string)
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
		labelTables[labelNames[strconv.Itoa(gl)]] = lts[gl]
	}

	// reading all domain embeddings
	domainSEmbs := make([][]string, 0)
	err = loadJson(DomainEmbsFile, &domainSEmbs)
	if err != nil {
		panic(err)
	}
	domainEmbs = make([][]float64, 0)
	domainEmbs = stringSlideToFloat(domainSEmbs)
	// reading table to emb id map
	tableEmbsMap = make(map[string][]int)
	err = loadJson(TableEmbsMap, &tableEmbsMap)
	if err != nil {
		panic(err)
	}
	// load the embedding of each label
	getLabelDomainEmbeddings()
	// eliminate labels without embeddings
	tm := make(map[string]bool)
	for _, gl := range labelIds {
		if _, ok := labelDomainEmbs[labelNames[strconv.Itoa(gl)]]; ok {
			labelsList = append(labelsList, labelNames[strconv.Itoa(gl)])
			labelTables[labelNames[strconv.Itoa(gl)]] = lts[gl]
			labels[labelNames[strconv.Itoa(gl)]] = true
			// adding tables of this label
			for _, t := range lts[gl] {
				if _, ok := tm[t]; !ok {
					tm[t] = true
				}
			}
		}
	}
	// making a list of all tables
	for t, _ := range tm {
		tables = append(tables, t)
	}
}

func getLabelDomainEmbeddings() {
	dim := len(domainEmbs[0])
	labelDomainEmbs = make(map[string][][]float64)
	for l, _ := range labels {
		lde := make([][]float64, 0)
		for _, t := range labelTables[l] {
			embIds := tableEmbsMap[t]
			for _, i := range embIds {
				// the first entry of an embedding slice is 0 and should be removed.
				lde = append(lde, domainEmbs[i][1:dim])
			}
		}
		labelDomainEmbs[l] = lde
		labelAvgEmb[l] = avg(lde)
	}
}

// evaluate the is-A quality of two label by computing the co-occurrence of
// labels for tables
func isAScorePlus(state1, state2 state) float64 {
	//log.Printf("cont: %f", getStatesCoverage(state1, state2))
	dkl := getKullbackLeibler(state1, state2)
	log.Printf("dkl: %f\n", dkl)
	return dkl
}

func getKullbackLeibler(state1, state2 state) float64 {
	d1 := make([][]float64, 0)
	d2 := make([][]float64, 0)
	for l1, _ := range state1.labels {
		d1 = append(d1, labelDomainEmbs[l1]...)
	}
	for l2, _ := range state2.labels {
		d2 = append(d2, labelDomainEmbs[l2]...)
	}
	return getNormalDKL(d1, d2)
}

// given a navigation path (consisting of ordered list of label sets), generate next
// possible states. Make sure the labels are not revisited and the semantic relevance is above a threshold.
func (navPath *path) generateNextStatePlus() bool {
	nextState := state{}
	// enumerate possible states
	allNextStates := enumerateAllNextStates(navPath)
	// should the path be terminated?
	if len(allNextStates) == 0 {
		log.Println("no next state")
		return false
	}
	// generate the prob distribution on possible states
	// compute semantic relevance of the current state to the potential next state: cosine of embs
	scores, _, cdf := generateNextStateDistributionPlus(allNextStates, navPath.states[len(navPath.states)-1])
	// should the path be terminated?
	maxScore := Max(scores)
	if maxScore < relevanceThreshold {
		log.Printf("terminating because sem rel is below the threshold: %f\n", maxScore)
		return false
	}
	// choose a state based on the distribution
	nextStateNotFound := true
	//checkedStateIds := make(map[int]bool)
	attemptCount := 0
	log.Printf("allNextStates: %d", len(allNextStates))
	for nextStateNotFound && attemptCount < 10 { //len(checkedStateIds) < len(allNextStates) && len(checkedStateIds) < 10 {
		attemptCount += 1
		newsid := pickAState(cdf)
		nextState = allNextStates[newsid]
		nextStateNotFound = !navPath.updatePath(nextState, scores[newsid])
	}
	return !nextStateNotFound //, navPath
}

// this method returns the CDF of the next state distribution
func generateNextStateDistributionPlus(states []state, prevState state) ([]float64, []float64, []float64) {
	rels := make([]float64, 0)
	ps := make([]float64, 0)
	cdf := make([]float64, 0)
	sum := 0.0
	for _, st := range states {
		r := isAScorePlus(st, prevState)
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

// simulate user navigation
func SimulatePlus() {
	for i := 0; i <= 20; i = len(seenPaths) {
		// pick a start state
		start := generateStartState()
		navPath := newPath(start)
		termination := false
		for ok := false; !ok; ok = termination {
			// generate next states and update path with the new state
			openend := navPath.generateNextStatePlus()
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
