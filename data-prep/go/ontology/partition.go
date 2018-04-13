package ontology

import (
	"io/ioutil"
	"encoding/json"
	"math"
	"sync"
)

var (
	sets = make(map[string]map[string]float64)
	universe = make(map[string]bool)
	weights = make(map[string]float64)
	coveredUniverse = make(map[string]bool)  
	uncoveredSets = make(map[string]map[string]float64)
	cost = 0.0
)

type setCost struct {
	name string
	cost float64
}

func preprocess() {
	tableLabelsProbs := make(map[string]map[string]float64)
	b, err := ioutil.ReadFile(TableLabelsProbs)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(b, &tableLabelsProbs)
	if err != nil {
		panic(err)
	}
	for table, labelProbs := range tableLabelsProbs {
		if _, ok := universe[table]; !ok {
			universe[table] = true
		}
		for label, prob := range labelProbs {
			if _, ok := sets[label]; !ok {
				sets[label] = make(map[string]float64)
				sets[label][table] = prob
				uncoveredSets[label] = make(map[string]float64)
				uncoveredSets[label][table] = prob
				weights[label] = prob
			} else { 
				sets[label][table] = prob
				uncoveredSets[label][table] = prob
				weights[label] += prob
			}
		}	
	}
} 

func GreedySetCover() {
	preprocess()
	for !coveredAllUniverse() {
		nextSet := pickSet()
		updateSets(nextSet)
	}
}

func updateSets(sc setCost) {
	for t, prob := range sets[sc.name] {
		coveredUniverse[t] = true
		// the cost might change
		cost += 1.0/prob	
	} 
	delete(uncoveredSets, sc.name)
}

func pickSet() setCost {
	setCosts := make(chan setCost)
	candSets := make(chan string)
	for l, _ := range uncoveredSets {
		candSets <- l
	}
	wg := &sync.WaitGroup{}
	wg.Add(20)
	for id := 0; id < 20; id++ {
		go func(id int) {
			for name := range candSets { 
				cost := computeUncoveredSetCostEffectiveness(name)
				setCosts <- setCost {
					name: name,
					cost: cost,
				}			
			}
			wg.Done()
		}(id)
	}           
	go func() {
		wg.Wait()
		close(setCosts)
	}()
	return pickMinCostSet(setCosts)
}

func coveredAllUniverse() bool {
	if len(coveredUniverse) == len(universe) {
		return true
	} 
	return false
}

func pickMinCostSet(setCosts <- chan setCost) setCost {
	var nextSet setCost 
	maxCost := math.MaxFloat64
	for sc := range setCosts {
		if sc.cost <= maxCost {
			nextSet = sc
		}		
	}
	return nextSet
}

func computeUncoveredSetCostEffectiveness(name string) float64{
	var cost float64
	var uncoveredCount int
	for t, p := range sets[name] {
		if coveredUniverse[t] == false {
			cost += 1.0/p
			uncoveredCount += 1
		}
	} 
	return cost / float64(uncoveredCount)
}
