//package ontology
package organization

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

type Organization struct {
	paths                    []path
	cognitiveComplexity      map[string][]float64
	transitionProbability    map[string]map[string][]float64
	expCognitiveComplexity   map[string]float64
	expTransitionProbability map[string]map[string]float64
	tableLabelSelectivity    map[string][]int
}

func NewOrganization() *Organization {
	return &Organization{
		paths:                    make([]path, 0),
		cognitiveComplexity:      make(map[string][]float64),
		transitionProbability:    make(map[string]map[string][]float64),
		expCognitiveComplexity:   make(map[string]float64),
		expTransitionProbability: make(map[string]map[string]float64),
		tableLabelSelectivity:    make(map[string][]int),
	}
}

func selectTable() string {
	// selecting an unseen table to generate a run for.
	count := 0
	for i := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tables)); count <= 2*len(tables); i = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tables)) {
		count += 1
		if _, ok := seenTables[i]; !ok {
			seenTables[i] = true
			return tables[i]
		}
	}
	return ""
}

// simulate user navigation
func (org *Organization) GenerateRuns(runNum int) {
	log.Printf("number of tables: %d", len(tables))
	for i := 0; i < runNum; i += 1 { //{i = len(org.paths) {
		tablename := tables[i]
		//log.Printf("%d tablename: %s", i, tablename)
		// pick an unseen table
		//tablename := selectTable()
		//if tablename == "" {
		//i -= 1
		//	continue
		//}
		// pick a start state
		// alternatively, we can have multiple start states
		start, found := generateSelectableStartState(tablename)
		if found == false {
			//log.Printf("could not find a selectable state for table %s.", tablename)
			continue
			//i -= 1
		}
		navPath := newMPath(start, tablename)
		indivisible := false
		for ok := false; !ok; ok = indivisible {
			// generate next states and update path with the new state
			nextStates := navPath.getNextStates()
			// finding the next state and computing the transition prob
			if len(nextStates) == 0 {
				indivisible = true
				continue
				//i -= 1
			}
			ns, np, nzStates := navPath.selectNextState(nextStates)
			if _, ok := org.tableLabelSelectivity[tablename]; !ok {
				org.tableLabelSelectivity[tablename] = make([]int, 0)
			}
			org.tableLabelSelectivity[tablename] = append(org.tableLabelSelectivity[tablename], len(nzStates))
			if len(nzStates) == 0 {
				indivisible = true
				continue
				//i -= 1
			}
			if _, ok := org.transitionProbability[navPath.states[len(navPath.states)-1].name]; !ok {
				org.transitionProbability[navPath.states[len(navPath.states)-1].name] = make(map[string][]float64)
			}
			org.transitionProbability[navPath.states[len(navPath.states)-1].name][ns.name] = append(org.transitionProbability[navPath.states[len(navPath.states)-1].name][ns.name], np)
			org.cognitiveComplexity[navPath.states[len(navPath.states)-1].name] = append(org.cognitiveComplexity[navPath.states[len(navPath.states)-1].name], getCognitiveComplexity(nzStates, len(navPath.states)))
			// updating path
			navPath.addState(ns, np)
		}
		if len(navPath.states) < 2 {
			//i -= 1
			continue
		}
		log.Printf("run %d", len(seenTables))
		seenTables[i] = true
		printTablePath(navPath)
		org.paths = append(org.paths, navPath)
	}
	log.Printf("seenTables: %d", len(seenTables))
}

func (org *Organization) ProcessRuns() {
	for s1, s2m := range org.transitionProbability {
		org.expTransitionProbability[s1] = make(map[string]float64)
		for s2, ps := range s2m {
			org.expTransitionProbability[s1][s2] = expectedValue(ps, 1.0/float64(len(org.paths)))
			//fmt.Printf("%s  to  %s : %f   %v\n", s1, s2, expectedValue(ps, 1.0/float64(len(org.paths))), ps)
		}
	}
	for s, cs := range org.cognitiveComplexity {
		org.expCognitiveComplexity[s] = expectedValue(cs, 1.0/float64(len(org.paths)))
		//fmt.Printf("complexity of %s : %f   %v\n", s, expectedValue(cs, 1.0/float64(len(org.paths))), cs)
	}
	dumpJson(TransitionProbabilityFile, &org.expTransitionProbability)
	dumpJson(StateProbabilityFile, &org.expCognitiveComplexity)
	dumpJson(TableLabelSelectivityFile, &org.tableLabelSelectivity)
}

func generateSelectableStartState(tablename string) (state, bool) {
	seenlis := make(map[int]bool)
	li := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(labelsList))
	label := labelsList[li]
	st := state{
		labels: map[string]bool{label: true},
		sem:    labelEmbs[label],
		tables: SliceToMap(labelTables[label]),
		name:   label,
	}
	for i := li; len(seenlis) < len(labelsList); i = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(labelsList)) {
		if _, ok := seenlis[i]; !ok {
			seenlis[i] = true
			label = labelsList[li]
			st = state{
				labels: map[string]bool{label: true},
				sem:    labelEmbs[label],
				tables: SliceToMap(labelTables[label]),
				name:   label,
			}
			if st.selectable(tablename) {
				return st, true
			}
		}
	}
	return st, false
}

func (p *path) selectNextState(nextStates []state) (state, float64, []state) {
	scores := make([]float64, 0)
	nzStates := make([]state, 0)
	correctStateId := -1
	maxScore := -1.0
	for i, st := range nextStates {
		p := getTransitionScore(p.states[len(p.states)-1], st, p.tablename)
		if p > 0.0 {
			nzStates = append(nzStates, st)
		}
		if p > maxScore {
			maxScore = p
			correctStateId = i
		}
		scores = append(scores, p)
	}
	probs := getSoftmax(scores)
	pr := rand.New(rand.NewSource(time.Now().UnixNano())).Float64()
	if pr < mistakeProb {
		incorrectStateId := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(labelsList))
		if incorrectStateId == correctStateId {
			incorrectStateId += 1
		}
		return nextStates[incorrectStateId], probs[incorrectStateId], nzStates
	}
	//log.Printf("len(scores): %d", len(scores))
	return nextStates[correctStateId], probs[correctStateId], nzStates
}

func getTransitionScore(state1, state2 state, tablename string) float64 {
	// for each domain in the table compute the emb similarity to the
	// state and return the best similarity.
	// normalize sim scores across all states.
	st2emb := labelAvgEmb[state1.name]
	embIds := tableEmbsMap[tablename]
	maxTransScore := 0.0
	dim := len(domainEmbs[0])
	for _, i := range embIds {
		// the first entry of an embedding slice is 0 and should be removed.
		de := domainEmbs[i][1:dim]
		tscore := Cosine(de, st2emb)
		if tscore > maxTransScore {
			maxTransScore = tscore
		}
	}
	return maxTransScore
}

func getCognitiveComplexity(nextStates []state, seenPathLen int) float64 {
	// delta - find states that lead to the table
	// cognitive complexity
	//return math.Exp(float64(len(nextStates)) / math.Max(float64(len(nextStates)), z))
	return math.Exp(float64(len(nextStates)) / float64(len(labels)-seenPathLen))
}

func (navPath *path) getNextStates() []state {
	states := make([]state, 0)
	// no label redundancy in a path
	for label, _ := range navPath.unseenLabels {
		sem := labelEmbs[label]
		st := state{
			labels:    map[string]bool{label: true},
			sem:       sem,
			tables:    SliceToMap(labelTables[label]),
			coherence: 0.0,
			name:      label,
		}
		if st.selectable(navPath.tablename) == true {
			//ts := getTransitionProbability(p.states[len(p.states)-1], st)
			//if tp > 0 {
			states = append(states, st)
			//	scores = append(scores, ts)
			//}
		}
	}
	return states
}

func (st *state) selectable(tablename string) bool {
	if _, ok := st.tables[tablename]; ok {
		return true
	}
	return false
}

func printTablePath(p path) {
	fmt.Printf("table: %s\n", p.tablename)
	for i, s := range p.states {
		ls := make([]string, 0)
		for k, _ := range s.labels {
			ls = append(ls, k)
		}
		fmt.Printf("state %d : %v (tables: %d) covered tables %d\n", i, ls, len(s.tables), len(p.coveredTables))
	}
	fmt.Println("----------------------------------")
}

func (p *path) addState(st state, prob float64) {
	// find the overlap of the tables covered by the path and those covered by the new state
	p.transitionProbs = append(p.transitionProbs, prob)
	if len(p.states) == 0 {
		p.probability = prob
	} else {
		p.probability *= prob
	}
	cts := make(map[string]bool)
	for t, _ := range p.coveredTables {
		if _, ok := st.tables[t]; ok {
			cts[t] = true
		}
	}
	p.coveredTables = cts
	p.states = append(p.states, st)
	for l, _ := range st.labels {
		p.seenLabels[l] = true
		delete(p.unseenLabels, l)
	}
}

func getSoftmax(scores []float64) []float64 {
	expSum := 0.0
	probs := make([]float64, 0)
	for _, v := range scores {
		expSum += math.Exp(v)
	}
	for _, v := range scores {
		probs = append(probs, v/expSum)
	}
	return probs
}

func newMPath(startState state, tablename string) path {
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
		tablename:       tablename,
	}
	return p
}
