package organization

import (
	"database/sql"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	. "github.com/RJMillerLab/opendata-ontology/go/embedding"
)

var (
	ftConn *FastText
)

type tag struct {
	name string
	sem  []float64
}

type space struct {
	states      map[string]state // state: tags, sem, id, datasets
	transitions map[string][]string
	stateIds    []string
	tagDatasets map[string][]string
	datasetEmbs map[string][]float64
	datasetTags map[string][]string
	tagSem      map[string][]float64
}

// datasetNum: total number of datasets
// perClassDatasetNum: max number of datasets for each class
func SynthesizeMetadata(numClass, datasetNum, perClassDatasetNum int) (map[string][]float64, map[string][]string, map[string][]float64, map[string][]string) {
	classes := selectClasses(numClass)
	log.Printf("generateTagsDatasetsFromClassesPlus")
	//tagSem, tagDatasets, datasetTags, datasetEmbs := generateTagsDatasetsFromClasses(classes, numClass, datasetNum)
	tagSem, tagDatasets, datasetTags, datasetEmbs := generateTagsDatasetsFromClassesPlus(classes, datasetNum, perClassDatasetNum)
	log.Printf("len(tagSem): %d  len(tagDatasets): %d  len(datasetTags): %d  len(datasetEmbs): %d", len(tagSem), len(tagDatasets), len(datasetTags), len(datasetEmbs))
	return tagSem, tagDatasets, datasetEmbs, datasetTags
}

func SynthesizeOrganizations(orgNum int, tagSem map[string][]float64, tagDatasets map[string][]string, datasetEmbs map[string][]float64, datasetTags map[string][]string) []organization {
	sp := generateStateSpace(tagSem, tagDatasets, datasetTags, datasetEmbs)
	orgs := sp.generateOrganizations(orgNum)
	return orgs
}

// finds classes with entities that are associated to more than one class
func selectClasses(numClass int) []string {
	db, err := sql.Open("sqlite3", YagoDBFile)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	//rows, err := db.Query(fmt.Sprintf(`select distinct class from benchmark order by random() limit %d;`, numClass))
	rows, err := db.Query(fmt.Sprintf(`select distinct category as class from (select distinct category from types where category like '%%English_%%' and  entity in (select      distinct entity from types  where category like '%%English_%%' group  by entity having count(distinct category)>1) group by category having count(distinct entity) > 500 order by random()  limit %d) union select * from (select distinct category from types where  category like '%%French_%%' and  entity in (select      distinct entity from types  where category like '%%French_%%' group  by entity having count(distinct category)>1) group by category having count(distinct entity) > 500 order by random() limit %d);`, numClass, numClass))
	//rows, err := db.Query("select distinct category from types where (category like '%%French_%%' or '%%English%%') and    entity in (select      distinct entity from types where category like '%%French_%%' or '%%English%%' " + fmt.Sprintf(`group  by entity having count(*)>1) order by random() limit %d;`, numClass))
	//rows, err := db.Query(fmt.Sprintf(`select distinct category from types where entity in (select entity from types group by entity having count(*)>1) limit %d;`, numClass))
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	classes := make([]string, 0)
	var class string
	for rows.Next() {
		err := rows.Scan(&class)
		if err != nil {
			log.Fatal(err)
		}
		classes = append(classes, class)
	}
	log.Printf("started with classes: %v", classes)
	return classes
}

func generateTagsDatasetsFromClassesPlus(classes []string, datasetNum, perClassDatasetNum int) (map[string][]float64, map[string][]string, map[string][]string, map[string][]float64) {
	tagDatasets := make(map[string][]string)
	datasetTags := make(map[string][]string)
	datasetEmbs := make(map[string][]float64)
	tagSem := make(map[string][]float64)
	datasets := make([]string, 0)
	for i := 0; i < datasetNum; i++ {
		datasets = append(datasets, "d_"+strconv.Itoa(i))
	}
	for _, c := range classes {
		values := getClassValues(c)
		sem, err := getSem(values)
		if err == nil {
			tagSem[c] = sem
			classDatasetNum := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(perClassDatasetNum-5) + 5
			tagDatasets[c] = make([]string, 0)
			for len(tagDatasets[c]) < classDatasetNum {
				did := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(datasetNum)
				if !containsStr(tagDatasets[c], datasets[did]) {
					tagDatasets[c] = append(tagDatasets[c], datasets[did])
					if _, ok := datasetTags[datasets[did]]; !ok {
						datasetTags[datasets[did]] = make([]string, 0)
					}
					datasetTags[datasets[did]] = append(datasetTags[datasets[did]], c)
				}
			}
		} else {
			log.Printf("no sem for this tag.")
		}
	}
	for _, dname := range datasets {
		d := generateDomainEmbsFromClasses(datasetTags[dname], dname)
		if len(d.sem) > 0 {
			datasetEmbs[d.name] = d.sem
		}
	}
	return tagSem, tagDatasets, datasetTags, datasetEmbs
}

func generateTagsDatasetsFromClasses(classes []string, numClass, datasetNum int) (map[string][]float64, map[string][]string, map[string][]string, map[string][]float64) {
	datasetNum = rand.New(rand.NewSource(time.Now().UnixNano())).Intn(datasetNum-10) + 10
	tagDatasets := make(map[string][]string)
	datasetTags := make(map[string][]string)
	datasetEmbs := make(map[string][]float64)
	tagSem := make(map[string][]float64)
	seen := make(map[string]bool)
	for len(tagSem) < numClass {
		i := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(classes))
		c := classes[i]
		if _, ok := seen[c]; ok {
			continue
		}
		seen[c] = true
		values := getClassValues(c)
		sem, err := getSem(values)
		if err == nil {
			ds := generateDomainEmbsFromClass(c, datasetNum)
			if len(ds) != 0 {
				tagSem[c] = sem
				tagDatasets[c] = make([]string, 0)
				for _, d := range ds {
					tagDatasets[c] = append(tagDatasets[c], d.name)
					if _, ok := datasetTags[d.name]; !ok {
						datasetTags[d.name] = make([]string, 0)
					}
					datasetTags[d.name] = append(datasetTags[d.name], c)
					datasetEmbs[d.name] = d.sem
				}
			}
		}
	}
	es := make([][]float64, 0)
	ds := make([]string, 0)
	for d, de := range datasetEmbs {
		ds = append(ds, d)
		es = append(es, de)
	}
	return tagSem, tagDatasets, datasetTags, datasetEmbs
}

// return entities of a class
func getClassValues(class string) []string {
	db, err := sql.Open("sqlite3", YagoDBFile)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	//rows, err := db.Query(fmt.Sprintf(`select distinct entity, category from types where category='%s';`, class))
	rows, err := db.Query(fmt.Sprintf(`select distinct entity, class from benchmark where class='%s';`, class))
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	values := make([]string, 0)
	var category string
	var entity string
	for rows.Next() {
		err := rows.Scan(&entity, &category)
		if err != nil {
			log.Fatal(err)
		}
		values = append(values, entity)
	}
	return values
}

// generate the semantic vector of a class using word embedding
func getSem(values []string) ([]float64, error) {
	sem, err := ftConn.GetDomainEmbMeanNoFreq(values)
	return sem, err
}

// returns the sem vector of a domain randomly generated from a set of classes.
// equal number of values are sampled from each class
func generateDomainEmbsFromClasses(classes []string, datasetname string) dataset {
	domain := make([]string, 0)
	dcount := 0
	for dcount < 5 {
		dcount += 1
		for _, class := range classes {
			values := getClassValues(class)
			card := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(values)-20) + 20
			// limiting the number of values in domains
			card = int(math.Min(float64(card), 100.0))
			domainm := make(map[string]bool)
			count := 0
			for len(domain) < card && count < (5*card) {
				einx := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(values))
				if _, ok := domainm[values[einx]]; !ok {
					domainm[values[einx]] = true
					domain = append(domain, values[einx])
				}
				count += 1
			}
		}
		dsem, err := getSem(domain)
		if err == nil {
			ds := dataset{
				name: datasetname,
				sem:  dsem,
			}
			return ds
		} else {
			log.Printf("dsem is nil")
		}
	}
	return dataset{}
}

// return the sem vector of a domain randomly generated from a class
func generateDomainEmbsFromClass(class string, domainNum int) []dataset {
	values := getClassValues(class)
	domains := make([]dataset, 0)
	count := 0
	seenDomStr := make(map[string]bool)
	for len(domains) < domainNum && count < 5*domainNum {
		card := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(values))
		// limiting the number of values in domains
		card = int(math.Min(float64(card), 100.0))
		domainm := make(map[string]bool)
		domain := make([]string, 0)
		domainStr := make([]string, 0)
		for len(domain) < card {
			einx := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(values))
			if _, ok := domainm[values[einx]]; !ok {
				domainm[values[einx]] = true
				domain = append(domain, values[einx])
				domainStr = append(domainStr, values[einx])
			}
		}
		sort.Strings(domainStr)
		dstr := ""
		for _, s := range domainStr {
			dstr += s
		}
		if _, ok := seenDomStr[dstr]; ok {
			continue
		}
		seenDomStr[dstr] = true
		count += 1
		domainSem, err := getSem(domain)
		if err == nil {
			ds := dataset{
				name: class + "_" + strconv.Itoa(len(domains)),
				sem:  domainSem,
			}
			domains = append(domains, ds)
		}
	}
	log.Printf("class %s len(domains): %d", class, len(domains))
	return domains
}

// returns all possible states that can be generated with a set of tags.
// power set of tags
func generateStateSpace(tagSem map[string][]float64, tagDatasets map[string][]string, datasetTags map[string][]string, datasetEmbs map[string][]float64) space {
	tags := getKeys(tagDatasets)
	stateTags := make([][]string, 0)
	// considering the tags with datasets
	tagsNum := len(tagDatasets)
	stateCount := int(math.Pow(2, float64(tagsNum)))
	log.Printf("raw stateCount: %d", stateCount)
	for i := 1; i < stateCount; i++ {
		s := make([]string, 0)
		bs := strconv.FormatUint(uint64(i), 2)
		ps := strings.SplitN(bs, "", -1)
		for j, p := range ps {
			pi, _ := strconv.Atoi(p)
			if pi == 1 {
				s = append(s, tags[len(ps)-j-1])
			}
		}
		// for now, discarding states with only one tag
		if len(s) < 2 {
			continue
		}
		stateTags = append(stateTags, s)
		if i%50 == 0 {
			log.Printf("generated %d states", i)
		}
	}
	log.Printf("number of states: %d", len(stateTags))
	states := make(map[string]state)
	for _, ts := range stateTags {
		sem, err := getSem(ts)
		if err == nil {
			st := state{
				tags:     ts,
				id:       "s" + strconv.Itoa(len(states)),
				sem:      sem,
				datasets: getTagDatasets(ts, tagDatasets),
			}
			states[st.id] = st
		} else {
			log.Printf("cannot generate sem for tags %s", ts)
		}
	}
	p := space{
		states:      states,
		stateIds:    getStateIds(states),
		tagDatasets: tagDatasets,
		datasetEmbs: datasetEmbs,
		datasetTags: datasetTags,
		tagSem:      tagSem,
	}
	p.generateTransitions()
	//log.Printf("transitions: %v", p.transitions)
	return p
}

func (p *space) generateTransitions() {
	stateids := getStateIds(p.states)
	transitions := make(map[string][]string)
	for i, sid1 := range stateids {
		s1 := p.states[sid1]
		for j := i + 1; j < len(stateids); j++ {
			sid2 := stateids[j]
			s2 := p.states[sid2]
			if isLinked(s1, s2) {
				if _, ok := transitions[sid1]; !ok {
					transitions[sid1] = make([]string, 0)
				}
				transitions[sid1] = append(transitions[sid1], sid2)
				if _, ok := transitions[sid2]; !ok {
					transitions[sid2] = make([]string, 0)
				}
				transitions[sid2] = append(transitions[sid2], sid1)
			}
		}
	}
	p.transitions = transitions
	//log.Printf("transitions: %v", transitions)
}

func (p *space) generateOrganizations(orgNum int) []organization {
	if orgNum < 0 {
		orgNum = len(p.transitions)
		log.Printf("Input orgNum is negative, generating %d orgs.", len(p.transitions))
	}
	orgs := make([]organization, 0)
	seenStarts := make(map[string]bool)
	log.Printf("len(p.transitions): %d", len(p.transitions))
	for len(orgs) < orgNum && len(seenStarts) < len(p.transitions) {
		start := p.randomStart()
		for seenStarts[start] {
			start = p.randomStart()
		}
		seenStarts[start] = true
		subgraph := p.traverseBreadthFirstSearch(start)
		org := p.subgraphToOrganization(subgraph, start, len(orgs))
		if org.isValid(len(p.tagDatasets), len(p.datasetTags)) {
			orgs = append(orgs, org)
		} else {
			log.Printf("org is not valid.")
		}
	}
	log.Printf("len(orgs): %d", len(orgs))
	return orgs
}

func (o *organization) isValid(tagNum, domainNum int) bool {
	tags := make(map[string]bool)
	for _, s := range o.states {
		for _, t := range s.tags {
			tags[t] = true
		}
	}
	return (len(o.reachables) == domainNum && len(tags) == tagNum)
}

func (p *space) subgraphToOrganization(graph map[string][]string, start string, id int) organization {
	states := make(map[string]state)
	for s, cs := range graph {
		states[s] = p.states[s]
		for _, c := range cs {
			states[c] = p.states[c]
		}
	}
	starts := []string{start}
	reachables := make(map[string]bool)
	for _, s := range states {
		for _, d := range s.datasets {
			reachables[d] = true
		}
	}
	return organization{
		states:      states,
		transitions: graph,
		starts:      starts,
		reachables:  reachables,
		id:          id,
	}
}

// breadth first search in the space DAG and returns a subgraph
func (p *space) traverseBreadthFirstSearch(start string) map[string][]string {
	// stop when all datasets and tags are covered.
	subgraph := make(map[string][]string)
	depths := make(map[string]int)
	nodes := make([]string, 0)
	nodes = append(nodes, start)
	depths[start] = 1
	sgdepth := 1
	seen := make(map[string]bool)
	ancestors := make(map[string][]string)
	for len(nodes) > 0 && len(seen) < len(p.transitions) {
		n := nodes[0]
		nodes = nodes[1:]
		seen[n] = true
		if _, ok := ancestors[n]; !ok {
			ancestors[n] = make([]string, 0)
		}
		cs := make([]string, 0)
		for _, c := range p.transitions[n] {
			seen[c] = true
			if containsStr(ancestors[n], c) {
				log.Printf("loop")
				continue
			}
			// creating a spanning DAG
			//	if _, ok := seen[c]; !ok {
			cs = append(cs, c)
			if _, ok := ancestors[c]; !ok {
				ancestors[c] = make([]string, 0)
			}
			if len(ancestors[n]) > 0 {
				ancestors[c] = append(ancestors[c], ancestors[n]...)
			}
			ancestors[c] = append(ancestors[c], n)
			//subgraph[n] = append(subgraph[n], c)
			nodes = append(nodes, c)
			if _, ok := depths[c]; !ok {
				depths[c] = depths[n] + 1
				sgdepth = int(math.Max(float64(sgdepth), float64(depths[c])))
			} else {
				depths[c] = int(math.Max(float64(depths[n]+1), float64(depths[c])))
				sgdepth = int(math.Max(float64(sgdepth), float64(depths[c])))
			}
			//	}
		}
		if len(cs) > 0 {
			if _, ok := subgraph[n]; !ok {
				subgraph[n] = cs
			} else {
				subgraph[n] = append(subgraph[n], cs...)
			}
		}
	}
	log.Printf("done ")
	//log.Printf("subgraph: %v", subgraph)
	return subgraph
}

// returns the id of a random start state
func (p *space) randomStart() string {
	sid := p.stateIds[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(p.transitions))]
	// restricting the number of search for start state
	//count := 0
	//for len(p.transitions[sid]) < 1 && count < 500 {
	//	sid = p.stateIds[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(p.transitions))]
	//	count += 1
	//}
	return sid
}

func isLinked(s1, s2 state) bool {
	// do no have tags overlap and do have dataset overlap
	return (!haveOverlap(s1.tags, s2.tags) && haveOverlap(s1.datasets, s2.datasets))
}

func getStateIds(m map[string]state) []string {
	keys := make([]string, 0)
	for k, _ := range m {
		keys = append(keys, k)
	}
	return keys
}

func Init() {
	ft, err := InitInMemoryFastText(FasttextDb, func(v string) []string {
		stopWords := []string{"ckan_topiccategory_", "ckan_keywords_", "ckan_tags_", "ckan_subject_", "socrata_domaincategory_", "socrata_domaintags_", "socrata_tags_"}
		for _, st := range stopWords {
			v = strings.Replace(v, st, "", -1)
		}
		v = strings.Replace(strings.Replace(strings.Replace(v, "_", " ", -1), "-", " ", -1), "\\'", " ", -1)
		return strings.Split(v, " ")
	}, func(v string) string {
		return strings.ToLower(strings.TrimFunc(strings.TrimSpace(v), unicode.IsPunct))
	})
	if err != nil {
		panic(err)
	}
	ftConn = ft
}

func getTagDatasets(tags []string, tagDatasets map[string][]string) []string {
	datasets := make(map[string]bool)
	for i := 0; i < len(tags); i++ {
		for _, d := range tagDatasets[tags[i]] {
			datasets[d] = true
		}
	}
	return mapToSlice(datasets)
}
