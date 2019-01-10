package hierarchy

import "log"

type organization struct {
	//states      []state
	stateIds     []int
	states       map[int]state
	transitions  map[int][]int
	root         state
	reachables   []dataset
	stateCounter int
}

type state struct {
	tags       []string
	sem        []float64
	population [][]float64
	label      []string
	id         int
	datasets   []string
	parents    []int
}

func (cing *clustering) ToOrganization() organization {
	log.Printf("ToOrganization")
	//states := make([]state, 0)
	stateIds := make([]int, 0)
	states := make(map[int]state)
	domains := make(map[string]bool)
	reachables := make([]dataset, 0)
	stateCounter := -1
	for _, c := range cing.clusters {
		s := state{
			tags:       make([]string, len(c.tags)),
			sem:        make([]float64, len(c.sem)),
			population: make([][]float64, len(c.population)),
			datasets:   getDatasets(c.tags),
			id:         c.id,
			parents:    make([]int, len(c.parents)),
		}
		copy(s.tags, c.tags)
		copy(s.sem, c.sem)
		copy(s.population, c.population)
		copy(s.parents, c.parents)
		if c.id > stateCounter {
			stateCounter = c.id
		}
		stateIds = append(stateIds, c.id)
		states[s.id] = s
		//states = append(states, s)
	}
	for _, t := range cing.root.tags {
		for _, domain := range tagDomains[t] {
			domains[domain] = true
		}
	}
	for dname, _ := range domains {
		d := dataset{
			name: dname,
			sem:  domainSems[dname],
		}
		reachables = append(reachables, d)
	}
	root := state{
		tags:       make([]string, len(cing.root.tags)),
		sem:        make([]float64, len(cing.root.sem)),
		population: make([][]float64, len(cing.root.population)),
		datasets:   getDatasets(cing.root.tags),
		id:         cing.root.id,
		parents:    make([]int, len(cing.root.parents)),
	}
	copy(root.tags, cing.root.tags)
	copy(root.sem, cing.root.sem)
	copy(root.population, cing.root.population)
	copy(root.parents, cing.root.parents)
	o := organization{
		states:       states,
		stateIds:     stateIds,
		transitions:  make(map[int][]int),
		root:         root,
		reachables:   reachables,
		stateCounter: stateCounter,
	}
	o.transitions = copymap(cing.hierarchy)
	return o
}

func getDatasets(tags []string) []string {
	ds := make([]string, 0)
	for _, t := range tags {
		for _, domain := range tagDomains[t] {
			ds = append(ds, domain)
		}
	}
	return ds
}

func (org organization) deepCopy() organization {
	states := make(map[int]state)
	stateIds := make([]int, 0)
	// deep copy of states
	for _, s := range org.states {
		t := state{
			tags:       make([]string, len(s.tags)),
			sem:        make([]float64, len(s.sem)),
			population: make([][]float64, len(s.population)),
			datasets:   make([]string, len(s.datasets)),
			id:         s.id,
			parents:    make([]int, len(s.parents)),
		}
		copy(t.tags, s.tags)
		copy(t.sem, s.sem)
		copy(t.population, s.population)
		copy(t.parents, s.parents)
		copy(t.datasets, s.datasets)
		stateIds = append(stateIds, s.id)
		states[s.id] = s
	}
	//deep copy of root state
	root := state{
		tags:       make([]string, len(org.root.tags)),
		sem:        make([]float64, len(org.root.sem)),
		population: make([][]float64, len(org.root.population)),
		datasets:   make([]string, len(org.root.datasets)),
		id:         org.root.id,
		parents:    make([]int, len(org.root.parents)),
	}
	copy(root.tags, org.root.tags)
	copy(root.sem, org.root.sem)
	copy(root.population, org.root.population)
	copy(root.parents, org.root.parents)
	copy(root.datasets, org.root.datasets)
	// deep copy of reachable datasets
	reachables := make([]dataset, 0)
	for _, d := range org.reachables {
		c := dataset{
			name: d.name,
			sem:  make([]float64, len(d.sem)),
		}
		copy(c.sem, d.sem)
		reachables = append(reachables, c)
	}
	return organization{
		states:      states,
		transitions: copymap(org.transitions),
		root:        root,
		reachables:  reachables,
		stateIds:    stateIds,
	}
}
