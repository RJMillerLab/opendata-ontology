package organization

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"sync"
	"time"
)

var (
	semDim = 300
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

func EvaluateOrganizations(organizations []organization, numRuns int) {
	log.Printf("EvaluateOrganizations")
	orgs := make(chan organization, 50)
	results := make(chan organization, 50)
	wg := &sync.WaitGroup{}
	fanout := 10
	wg.Add(fanout)
	go func() {
		for _, org := range organizations {
			orgs <- org
		}
	}()
	for i := 0; i < fanout; i++ {
		go func() {
			for org := range orgs {
				s := evaluateOrganization(org, numRuns)
				org.successProb = s
				results <- org
			}
			wg.Done()
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()
	log.Printf("saving evaluation results.")
	scoresOut := make([]float64, 0)
	orgsOut := make([]JOrganization, 0)
	top1Score := 0.0
	bottom1Score := 0.0
	top1Org := organization{}
	bottom1Org := organization{}
	for org := range results {
		log.Printf("output the result of org %d.", org.id)
		orgsOut = append(orgsOut, org.toJsonOrg())
		scoresOut = append(scoresOut, org.successProb)
		if top1Score < org.successProb {
			top1Score = org.successProb
			top1Org = org
		}
		if bottom1Score > org.successProb && org.successProb != 0 {
			bottom1Score = org.successProb
			bottom1Org = org
		}
		// processed all orgs
		if len(scoresOut) == len(organizations) {
			log.Printf("processed all orgs")
			close(orgs)
		}
	}
	log.Printf("Top-1 Org:")
	top1Org.Print()
	log.Printf("Bottom-1 Org:")
	bottom1Org.Print()
	dumpJson(OrgsEvaluationFile, &scoresOut)
	dumpJson(OrgsFile, &orgsOut)
}

func evaluateOrganization(org organization, numRuns int) float64 {
	log.Printf("evaluating org %d", org.id)
	runs := make(map[string][]run)
	totalRuns := 0
	datasetSuccessProbs := make(map[string]float64)
	orgSuccessProb := 0.0
	//count := 0
	for d, _ := range org.reachables {
		//	count += 1
		//	if count > 100 {
		//		continue
		//	}
		//log.Printf("domain: %s", d)
		rs := org.generateRuns(d, numRuns)
		totalRuns += len(rs)
		runs[d] = rs
		dsp := getDatasetSuccessProb(rs)
		datasetSuccessProbs[d] = dsp
		orgSuccessProb += dsp
	}
	orgSuccessProb = (1.0 / float64(len(org.reachables))) * orgSuccessProb
	log.Printf("org %d's success prob: %f with %d datasets and %d runs.", org.id, org.successProb, len(org.reachables), totalRuns)
	return orgSuccessProb
}

func getDatasetSuccessProb(runs []run) float64 {
	dsp := 0.0
	for _, r := range runs {
		//log.Printf("r.prob: %f  and r.success %f", r.prob, r.successProb)
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
	successCount := 0
	for i := 0; i < numRuns; i++ {
		dataset := newDataset(datasetname)
		start, startProb := org.selectStartState(dataset)
		query := newQuery()
		tags, tagProbs := org.selectTags(query, dataset, start)
		query.updateQuery(start, tags)
		//query := newQuery(start, tags)
		r := newRun(dataset, start, startProb, tags, tagProbs, query)
		stop := false
		currentState := start
		for !org.terminal(currentState) && !stop && !r.deadend() {
			next, nextProb := org.selectNextState(r)
			//	log.Printf("selectNextState")
			if nextProb == -1.0 {
				stop = true
				continue
			}
			currentState = next
			tags, tagProbs = org.selectTags(query, dataset, next)
			//	log.Printf("select tags")
			query.updateQuery(next, tags)
			r.updateRun(next, nextProb, tags, tagProbs, query)
			stop = false //doStop()
		}
		//r.prob = r.getQueryProbability()
		if !r.duplicate(runs) {
			if r.isSuccessful() {
				successCount += 1
				r.successProb = r.getTargetSelectionProbability()
				org.PrintRun(r)
			} else {
				r.successProb = 0.0
			}
			//fmt.Printf("org %d run %d: states %v successProb: %f #datasets in the end state %d\n", org.id, len(runs), r.states, r.successProb, len(r.datasets))
			//fmt.Println("-------------")
			runs = append(runs, r)
		}
	}
	log.Printf("org %d: %d runs out of %d are successful.", org.id, successCount, len(runs))
	return runs
}

//func newQuery(state string, tags []string) query {
func newQuery() query {
	q := query{
		tags:      make([]string, 0),
		stateTags: make(map[string][]string),
		sem:       make([]float64, semDim, semDim),
	}
	//if len(tags) > 0 {
	//	q.updateSem(tags)
	//}
	//q.stateTags[state] = tags
	return q
}

func (q *query) updateQuery(next string, tags []string) {
	// first update sem then add tags
	if len(tags) > 0 {
		q.updateSem(tags)
	}
	q.tags = append(q.tags, tags...)
	if _, ok := q.stateTags[next]; !ok {
		q.stateTags[next] = tags
	} else {
		q.stateTags[next] = append(q.stateTags[next], tags...)
	}
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

func (org organization) whichTags(s string, d dataset) []string {
	st := org.states[s]
	tags := st.tags
	goodTags := make([]string, 0)
	for _, t := range tags {
		if containsStr(tagDatasets[t], d.name) {
			goodTags = append(goodTags, t)
		}
	}
	return goodTags
}

func (org organization) selectTags(q query, d dataset, s string) ([]string, []float64) {
	//goodTags := org.whichTags(s, d)
	tags := org.states[s].tags
	ps := make([]float64, 0)
	denom := 0.0
	nums := make([]float64, 0)
	for _, t := range tags {
		f := math.Exp(-1.0 * norm(diff(d.sem, tagSem[t])))
		nums = append(nums, f)
		denom += f
	}
	for i, _ := range nums {
		ps = append(ps, nums[i]/denom)
	}
	// user selects random number (>0) of tags
	tnum := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(tags))
	if tnum == 0 {
		tnum += 1
	}
	sps, idxs := sortFloats(ps)
	stags := make([]string, 0)
	for i := 0; i < tnum; i += 1 {
		stags = append(stags, tags[idxs[i]])
	}
	//log.Printf("selected tags: %v   goodTags: %v", stags, goodTags)
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
	//parts := strings.Split(datasetname, "_")
	//i, _ := strconv.Atoi(parts[len(parts)-1])
	sem := datasetEmbs[datasetname] //datasetEmbs[i][1:]
	// for now, working with labels as dataset
	//sem := tagNameSem[datasetname]
	return sem
}

func newRun(d dataset, start string, startProb float64, tags []string, tagProbs []float64, q query) run {
	transitionProbs := make(map[int]map[int]float64)
	states := make([]string, 0)
	// state 0 is "null"
	states = append(states, "null")
	states = append(states, start)
	startId := 1
	transitionProbs[0] = make(map[int]float64)
	transitionProbs[0][startId] = startProb
	stateTagProbs := make(map[int]map[string]float64)
	stateTagProbs[startId] = make(map[string]float64)
	prob := startProb
	for i, t := range tags {
		stateTagProbs[startId][t] = tagProbs[i]
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
	unionProb := 0.0
	intersectProb := 1.0
	for i, t := range tags {
		r.tagProbs[nextId][t] = tagProbs[i]
		//r.prob *= tagProbs[i]
		unionProb += tagProbs[i]
		intersectProb *= tagProbs[i]
	}
	r.prob *= (unionProb - intersectProb)
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
	// the first state is always null
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
		log.Printf("stop()")
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
		log.Printf("terminal")
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
	//log.Printf("%d goodStates: %v", org.id, org.whichStates(nexts, r.target))
	ps := make(map[string]float64)
	denom := 0.0
	nums := make([]float64, 0)
	for _, n := range nexts {
		f := math.Exp(-1.0 * dot(diff(r.target.sem, r.query.sem), org.states[n].sem))
		nums = append(nums, f)
		denom += f
	}
	for i, s := range nexts {
		//	log.Printf("%d  %s : %f", org.id, s, nums[i]/denom)
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
	// adding the parent of the current node
	if len(r.states) > 2 {
		reachable = append(reachable, r.states[len(r.states)-2])
	}
	return reachable
}

func (org organization) whichStates(sts []string, d dataset) []string {
	goodStates := make([]string, 0)
	for _, s := range sts {
		if containsStr(org.states[s].datasets, d.name) {
			goodStates = append(goodStates, s)
		}
	}
	return goodStates
}

func InitSpace(tagDatasetsP map[string][]string, tagSemP, datasetEmbsP map[string][]float64) {
	tagDatasets = tagDatasetsP
	datasetEmbs = datasetEmbsP
	tagSem = tagSemP
}

func (org organization) toJsonOrg() JOrganization {
	jstates := make(map[string]JState)
	for _, s := range org.states {
		sj := JState{
			Tags:     s.tags,
			Id:       s.id,
			Datasets: s.datasets,
		}
		jstates[s.id] = sj
	}
	return JOrganization{
		States:      jstates,
		Transitions: org.transitions,
		Starts:      org.starts,
		Reachables:  org.reachables,
		SuccessProb: org.successProb,
	}
}

func (r run) deadend() bool {
	if len(r.datasets) == 0 {
		log.Printf("deadend")
		return true
	}
	return false
}

func (org organization) PrintRun(r run) {
	runStr := "org " + strconv.Itoa(org.id) + " : " + strconv.Itoa(len(r.states)) + " - " + strconv.FormatFloat(r.successProb, 'E', -1, 64) + " - " + "\n"
	for _, s := range r.states {
		st := org.states[s]
		runStr += st.id + " : "
		for _, t := range st.tags {
			runStr += t + " | "
		}
		runStr += "\n"
	}
	fmt.Println(runStr)
}
