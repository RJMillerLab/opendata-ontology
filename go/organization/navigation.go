package organization

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

type query struct {
	tags      []string            // all tags
	stateTags map[string][]string // tags selected in each state
	//states    []string            // state ids
	sem []float64
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

func evaluateOrganization(org organization) navigation {
	runs := make(map[string][]run)
	datasetSuccessProbs := make(map[string]float64)
	orgSuccessProb := 0.0
	for _, d := range tables {
		rs := org.generateRuns(d, numRuns)
		runs[d] = rs
		datasetSuccessProbs[d] = getDatasetSuccessProb(rs)
	}
	orgSuccessProb = (1.0 / float64(len(tables))) * orgSuccessProb
	return navigation{
		organization:        org,
		runs:                runs,
		datasetSuccessProbs: datasetSuccessProbs,
		orgSuccessProb:      orgSuccessProb,
	}
}

func getDatasetSuccessProb(runs []run) float64 {
	dsp := 0.0
	for _, r := range runs {
		dsp += r.successProb
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

// the probability of selecting a tag in a state, given a target
func (run run) tagSelectionProbability() float64 {
	return 0.0
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
		query := newQuery(tags) //start, tags)
		r := newRun(dataset, start, startProb, tags, tagProbs, query)
		stop := false
		currentState := start
		for !org.terminal(currentState) && !stop {
			next, nextProb := org.selectNextState(r)
			currentState = next
			tags, tagProbs = org.selectTags(dataset, next)
			query.updateQuery(next, tags)
			r.updateRun(next, nextProb, tags, tagProbs, query)
			stop = doStop()
		}
		//r.prob = r.getQueryProbability()
		if r.isSuccessful() {
			pTargetSelection := r.getTargetSelectionProbability()
			pRun := r.prob * pTargetSelection
			r.successProb = pRun
		} else {
			r.successProb = 0.0
		}
		if !r.duplicate(runs) {
			runs = append(runs, r)
		}
	}
	return runs
}

func newQuery(tags []string) query { //(start string, tags []string) query {
	//states := make([]string, 0)
	//states = append(states, start)
	//stateTags := make(map[int][]string)
	//stateTags[0] = tags
	q := query{
		tags: tags,
		//stateTags: stateTags,
		//	states:    states,
	}
	q.sem = q.updateSem(tags)
	return q
}

func (q query) updateQuery(next string, tags []string) {
	//q.states = append(q.states, next)
	q.tags = append(q.tags, next)
	q.stateTags[next] = tags
	q.updateSem(tags)
}

func (q query) updateSem(tags []string) []float64 {
	sems := make([][]float64, 0)
	if len(q.sem) > 0 {
		sems = append(sems, q.sem)
	}
	for _, t := range tags {
		sems = append(sems, labelAvgEmb[t])
	}
	return sum(sems)
}

func (org organization) selectTags(d dataset, s string) ([]string, []float64) {
	tags := org.states[s].tags
	ps := make([]float64, 0)
	denom := 0.0
	nums := make([]float64, 0)
	for _, t := range tags {
		f := math.Exp(norm(diff(d.sem, labelAvgEmb[t])))
		nums = append(nums, f)
		denom += f
	}
	for i, _ := range nums {
		ps = append(ps, nums[i]/denom)
	}
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
	sem := domainEmbs[i]
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
	datasets := q.evaluate()
	selectedTags := make(map[int][]string)
	selectedTags[startId] = tags
	return run{
		query:           q,
		datasets:        datasets,
		target:          d,
		states:          states,
		selectedTags:    selectedTags,
		transitionProbs: transitionProbs,
		tagProbs:        stateTagProbs,
		prob:            prob,
	}
}

func (r run) updateRun(next string, nextProb float64, tags []string, tagProbs []float64, q query) {
	r.query = q
	current := len(r.states) - 1
	r.states = append(r.states, next)
	nextId := len(r.states) - 1
	r.transitionProbs[current] = make(map[int]float64)
	r.transitionProbs[current][nextId] = nextProb
	r.selectedTags[nextId] = tags
	r.datasets = q.evaluate()
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

func (q query) evaluate() map[string]bool {
	datasets := make(map[string]bool)
	for i := 0; i < len(q.tags); i++ {
		if i == 0 {
			for _, d := range labelTables[q.tags[0]] {
				datasets[d] = true
			}
			continue
		}
		datasets = intersect(datasets, labelTables[q.tags[i]])
	}
	return datasets
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
	nexts := org.transitions[r.states[len(r.states)-1]]
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