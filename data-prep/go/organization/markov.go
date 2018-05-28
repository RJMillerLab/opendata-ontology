package ontology

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

var (
	cognitiveComplexity   map[string][]float64
	transitionProbability map[string]map[string][]float64
	mistakeProb           float64
	seenTables            map[int]bool
)

func selectTable() string {
	// selecting an unseen table to generate a run for.
	count := 0
	for i := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tables)); count <= len(tables); i = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tables)) {
		count += 1
		if _, ok := seenTables[i]; !ok {
			seenTables[i] = true
			return tables[i]
		}
	}
	return ""
}

// simulate user navigation
func GenerateRuns(runNum int) {
	for i := 0; i <= runNum; i = len(seenPaths) {
		// pick an unseen table
		tablename := selectTable()
		if tablename == "" {
			log.Printf("couldn't find an unseen table.")
			return
		}
		log.Printf("table: %s", tablename)
		// pick a start state
		start := generateStartState()
		navPath := newPath(start)
		indivisible := false
		for ok := false; !ok; ok = indivisible {
			// generate next states and update path with the new state
			nextStates := navPath.getNextStates()
			// finding the next state and computing the transition prob
			ns, np, nzStates := navPath.selectNextState(nextStates)
			if np < 0.0 {
				continue
			}
			transitionProbability[navPath.states[len(navPath.states)-1].name][ns.name] = append(transitionProbability[navPath.states[len(navPath.states)-1].name][ns.name], np)
			cognitiveComplexity[navPath.states[len(navPath.states)-1].name] = append(cognitiveComplexity[navPath.states[len(navPath.states)-1].name], getCognitiveComplexity(nzStates))
			if len(nzStates) == 0 {
				indivisible = true
			}
			// updating path
			navPath.addState(ns, np)
		}
		printTablePath(navPath)
	}
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
	return nextStates[correctStateId], probs[correctStateId], nzStates
}

func getTransitionScore(state1, state2 state, tablename string) float64 {
	// for each domain in the table compute the emb similarity to the
	// state and return the best similarity.
	// normalize sim scores across all states.
	return 1.0
}

func getCognitiveComplexity(nextStates []state) float64 {
	// delta - find states that lead to the table
	// cognitive complexity
	return math.Exp(float64(len(nextStates)) / math.Max(float64(len(nextStates)), z))
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
		if navPath.selectable(st) == true {
			//ts := getTransitionProbability(p.states[len(p.states)-1], st)
			//if tp > 0 {
			states = append(states, st)
			//	scores = append(scores, ts)
			//}
		}
	}
	return states
}

func (p *path) selectable(st state) bool {
	if _, ok := st.tables[p.tablename]; ok {
		return true
	}
	return false
}

func printTablePath(p path) {
	fmt.Printf("table: %s", p.tablename)
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
