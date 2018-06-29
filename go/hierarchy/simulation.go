package hierarchy

import (
	"log"
	"math"
)

type dataset struct {
	name string
	sem  []float64
}

type run struct {
	target          dataset                 // target dataset
	states          []int                   // the order of clusters visited during navigation
	transitionProbs map[int]map[int]float64 // cluster index -> cluster index -> prob
	prob            float64
	successProb     float64
}

func (org *organization) getNextStates(s state) []state {
	nextIds := org.transitions[s.id]
	nexts := make([]state, 0)
	for _, id := range nextIds {
		nexts = append(nexts, org.states[id])
	}
	return nexts
}

func newRun(start state, d dataset) run {
	return run{
		target:          d,
		states:          []int{start.id},
		transitionProbs: make(map[int]map[int]float64),
		prob:            1.0,
	}
}

func (r *run) updateRun(next state, nextProb float64) {
	//datasets        map[string]bool
	r.transitionProbs[r.states[len(r.states)-1]][next.id] = nextProb
	r.states = append(r.states, next.id)
	r.prob *= nextProb
}

func (org *organization) getNextProbabilities(s state, d dataset) map[int]float64 {
	nexts := org.getNextStates(s)
	probs := make(map[int]float64)
	expSum := 0.0
	for _, n := range nexts {
		probs[n.id] = getTransitionProb(n, d)
		expSum += math.Exp(probs[n.id])
	}
	for _, n := range nexts {
		probs[n.id] = math.Exp(probs[n.id]) / expSum
	}
	return probs
}

func getTransitionProb(c state, d dataset) float64 {
	return cosine(c.sem, d.sem)
}

func (org *organization) selectNextState(s state, d dataset) (state, float64) {
	nextProbs := org.getNextProbabilities(s, d)
	maxProb := -1.0
	bestNext := state{}
	for cid, tp := range nextProbs {
		if tp >= maxProb {
			maxProb = tp
			bestNext = org.states[cid]
		}
	}
	return bestNext, maxProb
}

func (org *organization) Evaluate() float64 {
	probSum := 0.0
	for _, d := range org.reachables {
		// work with log
		probSum += org.navigationSimulation(d)
	}
	successExpectation := probSum / float64(len(org.reachables))
	return successExpectation
}

func (org *organization) navigationSimulation(d dataset) float64 {
	current := org.root
	r := newRun(org.root, d)
	for !org.terminal(current) && !org.deadend(r) {
		next, nextProb := org.selectNextState(current, d)
		log.Printf("selectNextState: %d", next.id)
		if nextProb == -1.0 {
			continue
		}
		current = next
		r.updateRun(next, nextProb)
	}
	return r.prob
}

func (org *organization) navigationSuccessProb() float64 {
	return 0.0
}

func (org organization) terminal(s state) bool {
	if len(org.transitions[s.id]) == 0 {
		log.Printf("terminal")
		return true
	}
	return false
}

func (org organization) deadend(r run) bool {
	if containsStr(org.states[r.states[len(r.states)-1]].datasets, r.target.name) {
		return true
	}
	return false
}
