package organization

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

type query struct {
	tags      []string            // all tags
	stateTags map[string][]string // tags selected in each state
	sem       []float64
}

type dataset struct {
	name string
	sem  []float64
}

type run struct {
	query           query
	datasets        map[string]bool
	target          dataset  // target dataset
	states          []string // the order of states visited during navigation
	selectedTags    map[int][]string
	transitionProbs map[int]map[int]float64    // state index -> state index - > prob
	tagProbs        map[int]map[string]float64 // state index -> tag -> prob
	prob            float64
	successProb     float64
}

type navigation struct {
	organization        organization
	runs                map[string][]run
	datasetSuccessProbs map[string]float64
	orgSuccessProb      float64
}

func EvaluateOrganization(org organization, numRuns int) float64 {
	runs := make(map[string][]run)
	datasetSuccessProbs := make(map[string]float64)
	orgSuccessProb := 0.0
	count := 0
	for d, _ := range org.reachables {
		count += 1
		if count > 100 {
			continue
		}
		log.Printf("domain: %s", d)
		rs := org.generateRuns(d, numRuns)
		runs[d] = rs
		dsp := getDatasetSuccessProb(rs)
		datasetSuccessProbs[d] = dsp
		orgSuccessProb += dsp
	}
	orgSuccessProb = (1.0 / float64(len(org.reachables))) * orgSuccessProb
	if orgSuccessProb > 0.0 {
		log.Printf("orgSuccessProb is not zero.")
	}
	return orgSuccessProb
}

func getDatasetSuccessProb(runs []run) float64 {
	dsp := 0.0
	for _, r := range runs {
		dsp += (r.prob * r.successProb)
	}
	return dsp
}

// is target in the evaluation of query
func (r run) isSuccessful() bool {
	if r.datasets[r.target.name] {
		return true
	}
	return false
}

// state transition probability
func (org organization) tansitionProbability(state string) map[string]float64 {
	return make(map[string]float64)
}

func (org organization) generateRuns(datasetname string, numRuns int) []run {
	runs := make([]run, 0)
	for i := 0; i < numRuns; i++ {
		dataset := newDataset(datasetname)
		start, startProb := org.selectStartState(dataset)
		tags, tagProbs := org.selectTags(dataset, start)
		query := newQuery(start, tags)
		r := newRun(dataset, start, startProb, tags, tagProbs, query)
		stop := false
		currentState := start
		for !org.terminal(currentState) && !stop {
			next, nextProb := org.selectNextState(r)
			if nextProb == -1.0 {
				stop = true
				continue
			}
			currentState = next
			tags, tagProbs = org.selectTags(dataset, next)
			query.updateQuery(next, tags)
			r.updateRun(next, nextProb, tags, tagProbs, query)
			stop = doStop()
		}
		fmt.Printf("run %d: ", len(runs))
		fmt.Println(r.states)
		//fmt.Println(r.selectedTags)
		//r.prob = r.getQueryProbability()
		if r.isSuccessful() {
			log.Printf("issuccessful")
			r.successProb = r.getTargetSelectionProbability()
		} else {
			log.Printf("isnotsuccessful")
			r.successProb = 0.0
		}
		fmt.Printf("successProb: %f\n", r.successProb)
		fmt.Println("-------------")
		if !r.duplicate(runs) {
			runs = append(runs, r)
		}
	}
	return runs
}

func newQuery(state string, tags []string) query {
	q := query{
		tags:      tags,
		stateTags: make(map[string][]string),
	}
	if len(tags) > 0 {
		q.updateSem(tags)
	}
	q.stateTags[state] = tags
	return q
}

func (q *query) updateQuery(next string, tags []string) {
	q.tags = append(q.tags, next)
	if len(tags) > 0 {
		q.updateSem(tags)
	}
	q.stateTags[next] = tags
}

func (q *query) updateSem(tags []string) {
	sems := make([][]float64, 0)
	if len(q.sem) > 0 {
		sems = append(sems, q.sem)
	}
	for _, t := range tags {
		sems = append(sems, tagSem[t])
	}
	q.sem = updateAvg(q.sem, len(q.tags), sems)
}

func (q *query) updateSemPlus(tags []string) {
	sems := make([][]float64, 0)
	if len(q.sem) > 0 {
		sems = append(sems, q.sem)
	}
	for _, t := range tags {
		sems = append(sems, tagSem[t])
	}
	q.sem = sum(sems)
}

func (org organization) selectTags(d dataset, s string) ([]string, []float64) {
	tags := org.states[s].tags
	ps := make([]float64, 0)
	denom := 0.0
	nums := make([]float64, 0)
	for _, t := range tags {
		f := math.Exp(norm(diff(d.sem, tagSem[t])))
		nums = append(nums, f)
		denom += f
	}
	for i, _ := range nums {
		ps = append(ps, nums[i]/denom)
	}
	//log.Printf("tags ps: %v", ps)
	//log.Printf("tags: %v", tags)
	// user selects random number of tags
	tnum := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tags))
	sps, idxs := sort(ps)
	stags := make([]string, 0)
	for i := 0; i < tnum; i += 1 {
		stags = append(stags, tags[idxs[i]])
	}
	return stags, sps[:tnum]
}

func newDataset(datasetname string) dataset {
	return dataset{
		name: datasetname,
		sem:  getDatasetSem(datasetname),
	}
}

func getDatasetSem(datasetname string) []float64 {
	// when working with domains, datasetname is datasetname and
	// index of domain emb in tableEmbsMap, delimiter: '_'
	parts := strings.Split(datasetname, "_")
	i, _ := strconv.Atoi(parts[len(parts)-1])
	sem := domainEmbs[i][1:]
	// for now, working with labels as dataset
	//sem := tagNameSem[datasetname]
	return sem
}

func newRun(d dataset, start string, startProb float64, tags []string, tagProbs []float64, q query) run {
	transitionProbs := make(map[int]map[int]float64)
	states := make([]string, 0)
	// state 0 is "null"
	states = append(states, "null")
	transitionProbs[0] = make(map[int]float64)
	states = append(states, start)
	startId := len(states) - 1
	transitionProbs[0][startId] = startProb
	stateTagProbs := make(map[int]map[string]float64)
	stateTagProbs[startId] = make(map[string]float64)
	prob := startProb
	for i, _ := range tags {
		stateTagProbs[startId][tags[i]] = tagProbs[i]
		prob *= tagProbs[i]
	}
	selectedTags := make(map[int][]string)
	selectedTags[startId] = tags
	r := run{
		query:           q,
		target:          d,
		states:          states,
		selectedTags:    selectedTags,
		transitionProbs: transitionProbs,
		tagProbs:        stateTagProbs,
		prob:            prob,
	}
	r.evaluate()
	return r
}

func (r *run) updateRun(next string, nextProb float64, tags []string, tagProbs []float64, q query) {
	r.query = q
	currentId := len(r.states) - 1
	r.states = append(r.states, next)
	nextId := len(r.states) - 1
	r.transitionProbs[currentId] = make(map[int]float64)
	r.transitionProbs[currentId][nextId] = nextProb
	r.selectedTags[nextId] = tags
	r.evaluate()
	r.tagProbs[nextId] = make(map[string]float64)
	r.prob *= nextProb
	for i, _ := range tags {
		r.tagProbs[nextId][tags[i]] = tagProbs[i]
		r.prob *= tagProbs[i]
	}
}

func (r run) getTargetSelectionProbability() float64 {
	denom := 0.0
	nums := make(map[string]float64)
	for d, _ := range r.datasets {
		dsem := getDatasetSem(d)
		f := math.Exp(dot(r.target.sem, dsem))
		nums[d] = f
		denom += f
	}
	for d, p := range nums {
		if d == r.target.name {
			return p / denom
		}
	}
	return -1.0
}

func (r *run) evaluate() {
	s := r.states[len(r.states)-1]
	ts := r.query.stateTags[s]
	// the evaluation of the tags selected in a state is a disjunctive query
	ds := make(map[string]bool)
	for _, t := range ts {
		for _, d := range tagDatasets[t] {
			ds[d] = true
		}
	}
	// the evaluation of a query on a sequence of states in a conjunctive query
	if len(r.states) > 2 {
		r.datasets = intersect(ds, r.datasets)
	}
	// the first state is null
	if len(r.states) == 2 {
		r.datasets = ds
	}
}

func doStop() bool {
	r := rand.New(rand.NewSource(time.Now().UnixNano())).Float64()
	if r > stateStopProb {
		return true
	}
	return false
}

func (org organization) selectStartState(d dataset) (string, float64) {
	// starting from an empty state
	nexts := org.starts
	denom := 0.0
	nums := make([]float64, 0)
	for _, n := range nexts {
		f := math.Exp(dot(d.sem, org.states[n].sem))
		nums = append(nums, f)
		denom += f
	}
	start := ""
	maxProb := -1.0
	for i, s := range nexts {
		p := nums[i] / denom
		if p > maxProb {
			start = s
			maxProb = p
		}
	}
	return start, maxProb
}

func (org organization) terminal(s string) bool {
	if len(org.transitions[s]) == 0 {
		return true
	}
	return false
}

func (org organization) selectNextState(r run) (string, float64) {
	// TODO: adding backtracking to parent
	nextStateProbs := org.getTransitionProbabilities(r)
	// find the state with max prob
	nextState := ""
	maxProb := -1.0
	for s, p := range nextStateProbs {
		if p > maxProb {
			nextState = s
			maxProb = p
		}
	}
	return nextState, maxProb
}

func (org organization) getTransitionProbabilities(r run) map[string]float64 {
	nexts := org.reachableNextStates(r)
	//nexts := org.transitions[r.states[len(r.states)-1]]
	ps := make(map[string]float64)
	denom := 0.0
	nums := make([]float64, 0)
	for _, n := range nexts {
		f := math.Exp(dot(diff(r.target.sem, r.query.sem), org.states[n].sem))
		nums = append(nums, f)
		denom += f
	}
	for i, s := range nexts {
		ps[org.states[s].id] = nums[i] / denom
	}
	return ps
}

func (r run) getQueryProbability() float64 {
	p := 1.0
	for _, sps := range r.transitionProbs {
		for n, np := range sps {
			p *= np
			for _, tp := range r.tagProbs[n] {
				p *= tp
			}
		}
	}
	return p
}

func (r run) duplicate(runs []run) bool {
	for _, prevRun := range runs {
		if equalRuns(r, prevRun) {
			return true
		}
	}
	return false
}

func equalRuns(r1, r2 run) bool {
	if len(r1.states) != len(r2.states) {
		return false
	}
	for i, s := range r1.states {
		if s != r2.states[i] {
			return false
		}
		if len(r1.selectedTags[i]) != len(r2.selectedTags[i]) {
			return false
		}
		for _, t := range r1.selectedTags[i] {
			if !containsStr(r2.selectedTags[i], t) {
				return false
			}
		}
	}
	return true
}

func (org organization) reachableNextStates(r run) []string {
	reachable := make([]string, 0)
	s := r.states[len(r.states)-1]
	for _, n := range org.transitions[s] {
		if len(intersectPlus(r.datasets, org.states[n].datasets)) > 0 {
			reachable = append(reachable, n)
		}
	}
	if len(reachable) > 0 {
		fmt.Printf("reachable states from %s: %v\n", s, reachable)
	}
	return reachable
}
