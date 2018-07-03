package hierarchy

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/RJMillerLab/opendata-ontology/data-prep/go/pqueue"
)

var (
	coherence      map[int]float64
	coherenceQueue *pqueue.TopKQueue
)

// this operator changes the parent of a node to a random node
func (org organization) adopt() *organization {
	newOrg := org.deepCopy()
	child := org.stateIds[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))]
	for child == org.root.id {
		child = org.stateIds[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))]
	}
	parent := org.stateIds[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))]
	for parent == child || containsInt(org.transitions[parent], child) {
		parent = org.stateIds[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))]
	}
	newOrg.transitions[parent] = append(newOrg.transitions[parent], child)
	delParent := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states[child].parents))
	chs := newOrg.states[child]
	chs.parents = removeSlice(chs.parents, newOrg.states[child].parents[delParent])
	chs.parents = append(chs.parents, parent)
	newOrg.states[child] = chs
	log.Printf("child: %d parent: %d", child, parent)
	newOrg.updateAncestorsTags(child, parent)
	return newOrg
}

func (org *organization) updateAncestorsTags(child, parent int) {
	// update new and old parents
	toupdate := make(map[int]bool)
	toupdate[parent] = true
	next := parent
	queue := make([]int, 0)
	queue = append(queue, parent)
	for next != org.root.id && len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, p := range org.states[current].parents {
			if p == org.root.id {
				continue
			}
			if _, ok := toupdate[p]; !ok {
				queue = append(queue, p)
				toupdate[p] = true
			}
		}
	}
	for p, _ := range toupdate {
		pst := org.states[p]
		pst.tags = mergeset(org.states[p].tags, org.states[child].tags)
		org.states[p] = pst
	}
}

func (org *organization) split() {
	rsid, _ := coherenceQueue.Pop()
	sid := rsid.(int)
	for sid == org.root.id {
		rsid, _ = coherenceQueue.Pop()
		sid = rsid.(int)
	}
	tosplit := org.states[sid]
	delete(org.states, sid)
	split1, split2 := org.splitState(tosplit)
	org.states[split1.id] = split1
	org.states[split2.id] = split2
	// update parents
	for _, pid := range tosplit.parents {
		org.transitions[pid] = append(org.transitions[pid], split1.id)
		org.transitions[pid] = append(org.transitions[pid], split2.id)
	}
	// update children
	org.updateChildrenAfterSplit(tosplit, split1, split2)
}

func (org *organization) updateChildrenAfterSplit(parent, split1, split2 state) {
	org.transitions[split1.id] = make([]int, 0)
	org.transitions[split2.id] = make([]int, 0)
	for _, chid := range org.transitions[parent.id] {
		child := org.states[chid]
		for _, t := range child.tags {
			if containsStr(split1.tags, t) {
				org.transitions[split1.id] = append(org.transitions[split1.id], child.id)
				break
			}
			if containsStr(split2.tags, t) {
				org.transitions[split2.id] = append(org.transitions[split2.id], child.id)
				break
			}
		}
	}
	delete(org.transitions, parent.id)
}

func (org *organization) splitState(tosplit state) (state, state) {
	stags1, stags2 := splitStateTags(tosplit)
	st1 := org.splitStateByTags(stags1, tosplit, len(org.states))
	st2 := org.splitStateByTags(stags2, tosplit, len(org.states)+1)
	return st1, st2
}

func (org organization) splitStateByTags(tags []string, st state, sid int) state {
	population := make([][]float64, 0)
	sems := make([][]float64, 0)
	for _, t := range tags {
		sems = append(sems, tagSems[t])
		population = append(population, tagDomainSems[t]...)
	}
	return state{
		tags:       tags,
		population: population,
		sem:        avg(sems),
		id:         sid,
		parents:    st.parents,
		label:      make([]string, 0),
		datasets:   getDatasets(tags),
	}
}

func splitStateTags(pst state) ([]string, []string) {
	l1 := len(pst.tags) / 2.0
	//l2 := len(pst.tags) - l1
	mctags1 := make(map[string]bool)
	for len(mctags1) < l1 {
		tid := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(pst.tags))
		mctags1[pst.tags[tid]] = true
	}
	ctags1 := make([]string, 0)
	ctags2 := make([]string, 0)
	for t, _ := range mctags1 {
		ctags1 = append(ctags1, t)
	}
	for _, t := range pst.tags {
		if _, ok := mctags1[t]; !ok {
			ctags2 = append(ctags2, t)
		}
	}
	return ctags1, ctags2
}

func (org *organization) Optimize() *organization {
	oldOrg := org.deepCopy()
	oldOrgSuccessProb := oldOrg.Evaluate()
	newOrgSuccessProb := -1.0
	newOrg := &organization{}
	bestOrg := &organization{}
	bestOrgSuccessProb := -1.0
	bestAcceptProb := -1.0
	operators := []string{"adopt"}
	iteration := 30
	for i := 0; i < iteration; i++ {
		// the probability of a proposal for new org given an old org
		bestOpAcceptProb := -1.0
		bestOpOrg := &organization{}
		bestOrgOpSuccessProb := -1.0
		// apply all operators on org and evaluates
		for _, op := range operators {
			if op == "adopt" {
				newOrg = oldOrg.adopt()
			}
			newOrgSuccessProb = newOrg.Evaluate()
			log.Printf("newOrgSuccessProb: %f", newOrgSuccessProb)
			qOrgPrimeOrg := 1.0
			qOrgOrgPrime := 1.0
			acceptProb := math.Min(1.0, (newOrgSuccessProb*qOrgOrgPrime)/(oldOrgSuccessProb*qOrgPrimeOrg))
			log.Printf("acceptProb: %f", acceptProb)
			if acceptProb > bestOpAcceptProb {
				bestOpAcceptProb = acceptProb
				bestOpOrg = newOrg
				bestOrgOpSuccessProb = newOrgSuccessProb
			}
		}
		if bestOpAcceptProb > 0.0 {
			oldOrg = bestOpOrg
		} else {
			log.Printf("Cannot improve the org.")
		}
		if bestOrgOpSuccessProb > bestOrgSuccessProb {
			bestAcceptProb = bestOpAcceptProb
			bestOrg = bestOpOrg
			bestOrgSuccessProb = bestOrgOpSuccessProb
		}
		log.Printf("iteration: %d", i)
		log.Printf("bestOpAcceptProb: %f  bestOrgOpSuccessProb: %f", bestOpAcceptProb, bestOrgOpSuccessProb)
		log.Printf("-----------------")
	}
	log.Printf("oldOrgSuccessProb: %f  bestSuccessProb: %f  bestAcceptProb: %f", oldOrgSuccessProb, bestOrgSuccessProb, bestAcceptProb)
	return bestOrg
}

func initSearch(org organization) {
	coherenceQueue = pqueue.NewTopKQueue(len(org.states))
	coherence = org.computeStateCoherence()
	for sid, c := range coherence {
		coherenceQueue.Push(sid, c)
	}
}

func (org organization) computeStateCoherence() map[int]float64 {
	pairTagSims := computeTagPairSimilarity()
	coherence := make(map[int]float64)
	for _, s := range org.states {
		coherence[s.id] = 0.0
		for i, t1 := range s.tags {
			for j := i + 1; j < len(s.tags); j++ {
				t2 := s.tags[j]
				coherence[s.id] += pairTagSims[t1][t2]
			}
		}
	}
	return coherence
}

func computeTagPairSimilarity() map[string]map[string]float64 {
	pairTagSims := make(map[string]map[string]float64)
	for t1, sem1 := range tagSems {
		for t2, sem2 := range tagSems {
			if t1 == t2 {
				continue
			}
			if _, ok := pairTagSims[t1]; !ok {
				pairTagSims[t1] = make(map[string]float64)
			}
			if _, ok := pairTagSims[t2]; !ok {
				pairTagSims[t2] = make(map[string]float64)
			}
			sim := cosine(sem1, sem2)
			pairTagSims[t1][t2] = sim
			pairTagSims[t2][t1] = sim
		}
	}
	return pairTagSims
}
