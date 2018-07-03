package hierarchy

import (
	"log"
	"math"
	"math/rand"
	"time"
)

var (
	coherence map[int]float64
)

// this operator changes the parent of a node to a random node
func (org organization) adopt() *organization {
	newOrg := org.deepCopy()
	child := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))
	parent := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))
	for parent == child || containsInt(org.transitions[parent], child) {
		parent = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states))
	}
	newOrg.transitions[parent] = append(newOrg.transitions[parent], child)
	delParent := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(org.states[child].parents))
	newOrg.states[child].parents = removeSlice(newOrg.states[child].parents, newOrg.states[child].parents[delParent])
	newOrg.states[child].parents = append(newOrg.states[child].parents, parent)
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
	for next != org.root.id {
		current := queue[0]
		queue = queue[1:]
		for _, p := range org.states[current].parents {
			if p == org.root.id {
				continue
			}
			queue = append(queue, p)
			toupdate[p] = true
		}
	}
	for p, _ := range toupdate {
		org.states[p].tags = mergeset(org.states[p].tags, org.states[child].tags)
	}
}

//func split()

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
		if bestOpAcceptProb > bestAcceptProb {
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
	org.computeStateCoherence()
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
