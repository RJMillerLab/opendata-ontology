package hierarchy

import "log"

type organization struct {
	states      []state
	transitions map[int][]int
	root        state
	reachables  []dataset
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

func (cing *clustering) ToOrganization() *organization {
	log.Printf("ToOrganization")
	log.Printf("root: %d", cing.root.id)
	log.Printf("len(clusters): %d", len(cing.clusters))
	log.Printf("len(cing.transitions): %d", len(cing.hierarchy))
	states := make([]state, 0)
	reachables := make([]dataset, 0)
	for _, c := range cing.clusters {
		s := state{
			tags:       c.tags,
			sem:        c.sem,
			population: c.population,
			datasets:   getDatasets(c.tags),
			id:         c.id,
			parents:    c.parents,
		}
		states = append(states, s)
		for _, t := range c.tags {
			for _, domain := range tagDomains[t] {
				d := dataset{
					name: domain,
					sem:  domainSems[domain],
				}
				reachables = append(reachables, d)
			}
		}
	}
	root := state{
		tags:       cing.root.tags,
		sem:        cing.root.sem,
		population: cing.root.population,
		datasets:   getDatasets(cing.root.tags),
		id:         cing.root.id,
		parents:    cing.root.parents,
	}
	return &organization{
		states:      states,
		transitions: cing.hierarchy,
		root:        root,
		reachables:  reachables,
	}
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

func (org *organization) deepCopy() *organization {
	return &organization{
		states:      org.states,
		transitions: org.transitions,
		root:        org.root,
		reachables:  org.reachables,
	}
}
