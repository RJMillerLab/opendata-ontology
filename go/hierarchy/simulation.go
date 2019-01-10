package hierarchy

import (
	"fmt"
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
	if _, ok := r.transitionProbs[r.states[len(r.states)-1]]; !ok {
		r.transitionProbs[r.states[len(r.states)-1]] = make(map[int]float64)
	}
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
	//fmt.Printf("nextProbs: %v\n", nextProbs)
	return bestNext, maxProb
}

func (org organization) Evaluate() float64 {
	probSum := 0.0
	success := 0
	for _, d := range org.reachables {
		sp := org.simulateNavigation(d)
		probSum += sp
		if sp != 0.0 {
			success += 1
		}
	}
	successExpectation := probSum / float64(len(org.reachables))
	fmt.Printf("The success expectation of hierarchy navigation: %f (sum: %f) with %d reachables.\n", successExpectation, probSum, len(org.reachables))
	fmt.Printf("%d success out of %d.\n", success, len(org.reachables))
	return successExpectation
}

func (org *organization) simulateNavigation(d dataset) float64 {
	//fmt.Printf("datatset: %s: %v\n", d.name, domainTags[d.name])
	current := org.root
	r := newRun(org.root, d)
	for !org.terminal(current) && !org.deadend(r) {
		next, nextProb := org.selectNextState(current, d)
		//fmt.Printf("nextProb: %v  next state: %d\n", nextProb, next.id)
		if nextProb == -1.0 {
			continue
		}
		r.updateRun(next, nextProb)
		//if org.deadend(r) {
		//	fmt.Printf("deadend dataset: %v - run prob: %f at %d-th state\n", domainTags[d.name], r.prob, len(r.states))
		//	if org.terminal(next) {
		//		fmt.Printf("it is at terminal state.\n")
		//		for _, n := range org.transitions[current.id] {
		//			fmt.Printf("%d: %v\n", len(org.transitions[n]), org.states[n].tags)
		//		}
		//	} else {
		//		fmt.Printf("it is not at terminal state.\n")
		//	}
		//}
		//if org.terminal(next) {
		//	fmt.Printf("terminal - run prob: %f\n", r.prob)
		//}
		current = next
	}
	sp := org.getSuccessProb(r)
	if sp == 0.0 {
		//log.Printf("sp is zero")
	}
	//fmt.Printf("run's success prob: %f (run prob: %f)\n", sp, r.prob)
	//fmt.Printf("------------------------\n")
	return sp
}

func (org *organization) getSuccessProb(r run) float64 {
	if !containsStr(org.states[r.states[len(r.states)-1]].datasets, r.target.name) {
		return 0.0
	}
	// add the prob of dataset selection in the last state
	return r.prob
}

func (org organization) terminal(s state) bool {
	if len(org.transitions[s.id]) == 0 {
		//fmt.Printf("reached a leaf state in hierarchy\n")
		return true
	}
	return false
}

func (org organization) deadend(r run) bool {
	if containsStr(org.states[r.states[len(r.states)-1]].datasets, r.target.name) {
		return false
	}
	//fmt.Printf("deadend state: target is not reachable\n")
	return true
}
