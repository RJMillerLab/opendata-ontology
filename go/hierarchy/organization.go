package hierarchy

type organization struct {
	states      []state
	transitions map[int][]int
	root        state
}

type state struct {
	tags       []string
	sem        []float64
	population [][]float64
	label      []string
	id         int
	datasets   []string
}

func (cing *clustering) toOrganization() organization {
	states := make([]state, 0)
	for _, c := range cing.clusters {
		s := state{
			tags:       c.tags,
			sem:        c.sem,
			population: c.population,
			datasets:   getDatasets(c.tags),
		}
		states = append(states, s)
	}
	root := state{
		tags:       cing.root.tags,
		sem:        cing.root.sem,
		population: cing.root.population,
		datasets:   getDatasets(cing.root.tags),
	}
	return organization{
		states:      states,
		transitions: cing.hierarchy,
		root:        root,
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
