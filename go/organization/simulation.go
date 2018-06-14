package organization

import (
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"time"
)

type node struct {
	tags  []string
	id    string
	depth int
	index int
}

var (
	overlappingTags map[string][]string
)

func GenerateOrganizations(orgNum int) []organization {
	ODTransitions()
	log.Printf("generating %d orgs.", orgNum)
	orgs := make([]organization, 0)
	for len(orgs) < orgNum {
		org := generateRandomOrganization()
		orgs = append(orgs, org)
		fmt.Printf("Org %d:\n", len(orgs))
		//for _, s := range org.states {
		//	fmt.Printf("%s: %v\n", s.id, s.tags)
		//}
		fmt.Println("------------")
		fmt.Println(org.transitions)
		fmt.Println("------------")
		fmt.Println(org.starts)
	}
	return orgs
}

func generateRandomOrganization() organization {
	numStarts := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(5) + 1
	starts := make([]string, 0)
	seenStates := make([]map[string]bool, 0)
	expandableLeaves := make([]node, 0)
	nodes := make([]node, 0)
	edges := make(map[string][]string)
	for len(starts) < numStarts {
		stateTags := generateRandomStart()
		if !duplicateState(stateTags, seenStates) {
			stateId := "s" + strconv.Itoa(len(nodes))
			//log.Printf("Not a duplicates. Assigned %s to state.", stateId)
			n := node{
				tags:  mapToSlice(stateTags),
				depth: 1,
				id:    stateId,
				index: len(nodes),
			}
			starts = append(starts, stateId)
			nodes = append(nodes, n)
			seenStates = append(seenStates, stateTags)
			expandableLeaves = append(expandableLeaves, n)
		}
	}
	// expand all nodes for the first time
	i := 0
	for len(expandableLeaves) > 0 {
		n := expandableLeaves[i]
		//log.Printf("len(expandableLeaves): %d", len(expandableLeaves))
		shouldExpand := n.expand()
		isStartNode := containsStr(starts, n.id)
		if shouldExpand || isStartNode {
			childrenNum := 0
			if isStartNode {
				childrenNum = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(4) + 1
			} else {
				childrenNum = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(5)
			}
			//log.Printf("generating %d children for %s.", childrenNum, n.id)
			for i := 0; i < childrenNum; i++ {
				stateTags := generateRandomState(n)
				if !duplicateState(stateTags, seenStates) {
					stateId := "s" + strconv.Itoa(len(nodes))
					//log.Printf("Not a duplicates. Assigned %s to state.", stateId)
					c := node{
						id:    stateId,
						tags:  mapToSlice(stateTags),
						depth: n.depth + 1,
						index: len(nodes),
					}
					nodes = append(nodes, c)
					if _, ok := edges[n.id]; !ok {
						edges[n.id] = make([]string, 0)
					}
					edges[n.id] = append(edges[n.id], c.id)
					expandableLeaves = append(expandableLeaves, c)
				}
			}
			nodes[n.index] = n
		}
		//else {
		//	log.Printf("should not expand %s", n.id)
		//}
		if len(expandableLeaves) == 1 {
			expandableLeaves = make([]node, 0)
		} else {
			expandableLeaves = append(expandableLeaves[:i], expandableLeaves[i+1:]...)
		}
		//log.Printf("len(expandableLeaves): %d", len(expandableLeaves))
	}
	//log.Printf("There is %d nodes in the list.", len(expandableLeaves))
	return organization{
		states:      nodesToStates(nodes),
		transitions: edges,
		starts:      starts,
	}
}

func nodesToStates(ns []node) map[string]state {
	states := make(map[string]state)
	for _, n := range ns {
		s := state{
			tags:     n.tags,
			id:       n.id,
			sem:      getSem(n.tags),
			datasets: getDatasets(n.tags),
		}
		states[n.id] = s
	}
	return states
}

func (n node) expand() bool {
	if n.depth > 5 {
		return false
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano())).Float64()
	if r > 0.5 {
		return true
	}
	return false
}

func ODTransitions() {
	overlappingTags = make(map[string][]string)
	seenPairs := make(map[string]bool)
	overlapCount := 0
	pairCount := 0
	log.Printf("tagDatasets: %d", len(tagDatasets))
	for t1, ds1 := range tagDatasets {
		for t2, ds2 := range tagDatasets {
			if t1 != t2 && !seenPairs[t1+"@"+t2] && !seenPairs[t2+"@"+t1] {
				pairCount += 1
				seenPairs[t1+"@"+t2] = true
				seenPairs[t2+"@"+t1] = true
				if haveOverlap(ds1, ds2) {
					overlapCount += 1
					if _, ok := overlappingTags[t1]; !ok {
						overlappingTags[t1] = make([]string, 0)
					}
					overlappingTags[t1] = append(overlappingTags[t1], t2)
					if _, ok := overlappingTags[t2]; !ok {
						overlappingTags[t2] = make([]string, 0)
					}
					overlappingTags[t2] = append(overlappingTags[t2], t1)
				}
			}
		}
	}
	log.Printf("number of pairs: %d", pairCount)
	log.Printf("number of pairs with overlap: %d", overlapCount)
}

func generateRandomStart() map[string]bool {
	// each state consists of up to 10% of tags)
	tags := make(map[string]bool)
	// each state has at least one tag
	tagNum := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(int(0.1*float64(len(overlappingTags)))) + 1
	for len(tags) < tagNum {
		t := labelsList[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(overlappingTags))]
		if len(overlappingTags[t]) > 0 {
			tags[t] = true
		}
	}
	return tags
}

func generateRandomState(n node) map[string]bool {
	options := make(map[string]bool)
	for _, t := range n.tags {
		for _, r := range overlappingTags[t] {
			options[r] = true
		}
	}
	tags := make(map[string]bool)
	// each state has at least one tag
	tagNum := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(options)) + 1
	for len(tags) < tagNum {
		t := labelsList[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tagSem))]
		tags[t] = true
	}
	return tags
}

func duplicateState(tags map[string]bool, seenStates []map[string]bool) bool {
	for _, st := range seenStates {
		seen := true
		for t, _ := range tags {
			if _, ok := st[t]; !ok {
				seen = false
				break
			}
		}
		if seen {
			return true
		}
	}
	return false
}

func getSem(tags []string) []float64 {
	sems := make([][]float64, 0)
	for _, t := range tags {
		sems = append(sems, tagSem[t])
	}
	return sum(sems)
}

func getDatasets(tags []string) []string {
	datasets := make(map[string]bool)
	for i := 0; i < len(tags); i++ {
		if i == 0 {
			for _, d := range tagDatasets[tags[0]] {
				datasets[d] = true
			}
			continue
		}
		datasets = intersectPlus(datasets, tagDatasets[tags[i]])
	}
	return mapToSlice(datasets)
}
