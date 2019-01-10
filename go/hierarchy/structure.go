package hierarchy

import (
	"log"
	"math/rand"
	"time"

	"github.com/RJMillerLab/opendata-ontology/data-prep/go/pqueue"
)

var (
	stateCoherence map[int]float64
	coherenceQueue *pqueue.TopKQueue
	pairTagSims    map[string]map[string]float64
	stateCounter   int
)

// this operator changes the parent of a node to a random node
func (org organization) adopt() organization {
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
	log.Printf("child: %d len(org.states[child].parents): %d len(org.transitions[child]): %d", child, len(org.states[child].parents), len(org.transitions[child]))
	delParent := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states[child].parents))
	log.Printf("org.transitions[org.stateIds[newOrg.states[child].parents[delParent]]]: %d", org.transitions[org.stateIds[newOrg.states[child].parents[delParent]]])
	org.transitions[org.stateIds[newOrg.states[child].parents[delParent]]] = removeSlice(org.transitions[org.stateIds[newOrg.states[child].parents[delParent]]], child)
	chs := newOrg.states[child]
	chs.parents = removeSlice(chs.parents, newOrg.states[child].parents[delParent])
	chs.parents = append(chs.parents, parent)
	newOrg.states[child] = chs
	//log.Printf("child: %d parent: %d", child, parent)
	newOrg.updateAncestorsTags(child, parent)
	return newOrg
}

func (org organization) updateAncestorsTags(child, parent int) {
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

func (org organization) split() organization {
	newOrg := org.deepCopy()
	rsid, c := coherenceQueue.Pop()
	sid := rsid.(int)
	for sid == org.root.id || len(org.states[sid].tags) < 2 {
		if _, ok := org.states[sid]; !ok {
			continue
		}
		if coherenceQueue.Empty() {
			sid = -1
			log.Printf("empty coherence queue: could not find state to split.")
			break
		}
		rsid, c = coherenceQueue.Pop()
		sid = rsid.(int)
	}
	if sid == -1 {
		log.Printf("no split can be done.")
		return newOrg
	}
	log.Printf("splitting %d with coherence %f", sid, c)
	tosplit := newOrg.states[sid]
	split1, split2 := newOrg.splitState(tosplit)
	newOrg.states[split1.id] = split1
	newOrg.states[split2.id] = split2
	delete(newOrg.states, sid)
	// update children
	newOrg.updateChildrenAfterSplit(tosplit, split1, split2)
	delete(newOrg.transitions, sid)
	// update parents
	for _, pid := range tosplit.parents {
		newOrg.transitions[pid] = append(newOrg.transitions[pid], split1.id)
		newOrg.transitions[pid] = append(newOrg.transitions[pid], split2.id)
		newOrg.transitions[pid] = removeSlice(newOrg.transitions[pid], tosplit.id)
	}
	// TODO: uncomment this
	// update coherence queue
	//split1Coh := org.getStateCoherence(split1)
	//split2Coh := org.getStateCoherence(split2)
	//coherenceQueue.Push(split1.id, split1Coh)
	//coherenceQueue.Push(split2.id, split2Coh)
	log.Printf("split: len(newOrg): %d (old: %d)", len(newOrg.states), len(org.states))
	return newOrg
}

func (org organization) updateChildrenAfterSplit(tosplit, split1, split2 state) {
	org.transitions[split1.id] = make([]int, 0)
	org.transitions[split2.id] = make([]int, 0)
	for _, chid := range org.transitions[tosplit.id] {
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
}

func (org organization) splitState(tosplit state) (state, state) {
	stags1, stags2 := splitStateTags(tosplit)
	st1 := org.splitStateByTags(stags1, tosplit)
	//log.Printf("st1: %d", len(st1.sem))
	st2 := org.splitStateByTags(stags2, tosplit)
	//log.Printf("st2: %d", len(st2.datasets))
	return st1, st2
}

func (org organization) splitStateByTags(tags []string, st state) state {
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
		id:         getNewStateId(),
		parents:    st.parents,
		label:      make([]string, 0),
		datasets:   getDatasets(tags),
	}
}

func splitStateTags(pst state) ([]string, []string) {
	l1 := len(pst.tags) / 2.0
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

func (org organization) Optimize() organization {
	initSearch(org)
	log.Printf("starting opt with %d states.", len(org.states))
	oldOrg := org.deepCopy()
	oldOrgSuccessProb := oldOrg.Evaluate()
	newOrgSuccessProb := -1.0
	newOrg := organization{}
	bestOrg := org.deepCopy()
	bestOrgSuccessProb := oldOrgSuccessProb
	bestAcceptProb := oldOrgSuccessProb
	operators := []string{"adopt"}     //"split"
	iteration := coherenceQueue.Size() // 30
	log.Printf("coherenceQueue: %d", coherenceQueue.Size())
	for i := 0; i < iteration; i++ {
		log.Printf("iteration: %d", i)
		// TODO: remove this for continuous search
		log.Printf("len(org.states): %d", len(org.states))
		oldOrg = org.deepCopy()
		// the probability of a proposal for new org given an old org
		bestOpAcceptProb := -1.0
		bestOpOrg := organization{}
		bestOrgOpSuccessProb := -1.0
		// apply all operators on org and evaluates
		for _, op := range operators {
			if op == "adopt" {
				log.Printf("adopt")
				newOrg = oldOrg.adopt()
			} else if op == "split" {
				log.Printf("split")
				if coherenceQueue.Size() == 0 {
					log.Println("empty coherence queue.")
					continue
				}
				newOrg = oldOrg.split()
			} else {
				continue
			}
			newOrgSuccessProb = newOrg.Evaluate()
			log.Printf("newOrgSuccessProb with op: %f", newOrgSuccessProb)
			acceptProb := newOrgSuccessProb
			if acceptProb > bestOpAcceptProb {
				bestOpAcceptProb = acceptProb
				bestOpOrg = newOrg.deepCopy()
				bestOrgOpSuccessProb = newOrgSuccessProb
			}
			log.Printf("op: %s - num of states: new %d old %d (input: %d)", op, len(newOrg.states), len(oldOrg.states), len(org.states))
			//oldOrg = newOrg.deepCopy()
		}
		//if bestOpAcceptProb > 0.0 {
		//	oldOrg = bestOpOrg
		//}
		if bestOrgOpSuccessProb > bestOrgSuccessProb {
			log.Printf("improved the org from %f to %f.", bestOrgSuccessProb, bestOrgOpSuccessProb)
			bestAcceptProb = bestOpAcceptProb
			bestOrg = bestOpOrg.deepCopy()
			bestOrgSuccessProb = bestOrgOpSuccessProb
		}
		log.Printf("bestOpAcceptProb: %f  bestOrgOpSuccessProb: %f", bestOpAcceptProb, bestOrgOpSuccessProb)
		log.Printf("num of states: new %d old %d (input: %d)", len(newOrg.states), len(oldOrg.states), len(org.states))
		log.Printf("-----------------")
		if bestOrgSuccessProb == 1.0 {
			break
		}
	}
	log.Printf("oldOrgSuccessProb: %f  bestSuccessProb: %f  bestAcceptProb: %f best org num states: %d (input: %d)", oldOrgSuccessProb, bestOrgSuccessProb, bestAcceptProb, len(bestOrg.states), len(org.states))
	return bestOrg
}

func initSearch(org organization) {
	stateCounter = org.stateCounter
	coherenceQueue = pqueue.NewTopKQueue(len(org.states))
	stateCoherence := org.getStatesCoherence()
	for sid, c := range stateCoherence {
		coherenceQueue.Push(sid, c)
	}
}

func (org organization) getStatesCoherence() map[int]float64 {
	pairTagSims = computeTagPairSimilarity()
	scs := make(map[int]float64)
	for _, s := range org.states {
		// including tag numbers
		scs[s.id] = org.getStateCoherence(s) / float64(len(s.tags))
	}
	return scs
}

func (org organization) getStateCoherence(s state) float64 {
	coherence := 0.0
	for i, t1 := range s.tags {
		for j := i + 1; j < len(s.tags); j++ {
			t2 := s.tags[j]
			coherence += pairTagSims[t1][t2]
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

func getNewStateId() int {
	stateCounter += 1
	return stateCounter
}
