package organization

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode"

	. "github.com/RJMillerLab/opendata-ontology/go/embedding"
)

var (
	labels          map[string]bool
	tables          []string
	labelNames      map[string]string
	labelsList      []string
	labelDomainEmbs map[string][][]float64
	tagNameEmb      map[string][]float64
	tableEmbsMap    map[string][]int
	tagDatasets     map[string][]string
	domainEmbs      [][]float64

	domains       []string
	tagSem        map[string][]float64
	tagNameSem    map[string][]float64
	tagDomains    map[string][]string
	startStateNum = 5
	stateStopProb = 0.5
)

type state struct {
	tags     []string
	sem      []float64
	id       string
	datasets []string
}

type organization struct {
	states      map[string]state    // mapping from id to state
	transitions map[string][]string // state transitions
	starts      []string            // a list of state ids
	reachables  map[string]bool     // reachable datasets by this organization
}

func ReadOrganization() organization {
	f, err := os.Open(OrgFile)
	fmt.Printf("OrgFile: %s\n", OrgFile)
	defer f.Close()
	if err != nil {
		panic(err)
	}
	states := make(map[string]state)
	transitions := make(map[string][]string)
	scanner := bufio.NewScanner(f)
	// the first line is "states" followed by lines containing state id and tags separated by '|'
	scanner.Scan()
	line := scanner.Text()
	for scanner.Scan() {
		line = strings.TrimSpace(scanner.Text())
		if line == "transitions" {
			break
		}
		parts := strings.Split(line, "|")
		id := parts[0]
		tags := parts[1:]
		sem := getStateSem(tags)
		datasets := make([]string, 0)
		for _, t := range tags {
			datasets = append(datasets, tagDomains[t]...)
		}
		s := state{
			id:       id,
			tags:     tags,
			sem:      sem,
			datasets: datasets,
		}
		states[id] = s
	}
	// reading transitions
	for scanner.Scan() {
		line = strings.TrimSpace(scanner.Text())
		if line == "start states" {
			break
		}
		parts := strings.SplitN(line, "|", 2)
		sid1 := parts[0]
		sid2 := parts[1]
		if _, ok := transitions[sid1]; !ok {
			transitions[sid1] = make([]string, 0)
		}
		transitions[sid1] = append(transitions[sid1], sid2)
	}
	org := organization{
		states:      states,
		transitions: transitions,
	}

	// reading start states
	starts := make([]string, 0)
	if scanner.Scan() {
		line = strings.TrimSpace(scanner.Text())
		starts = strings.Split(line, "|")
	} else {
		starts = org.generateSourceStates()
	}
	org.starts = starts
	return org
}

func (org organization) generateSourceStates() []string {
	starts := make([]string, 0)
	for len(starts) != startStateNum {
		id := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(startStateNum)
		if !containsStr(starts, org.starts[id]) {
			starts = append(starts, org.starts[id])
		}
	}
	return starts
}

func getStateSem(tags []string) []float64 {
	vecs := make([][]float64, 0)
	sem := make([]float64, 0)
	for _, t := range tags {
		if _, ok := tagSem[t]; !ok {
			log.Printf("Semantics of tag %s is undefined.", t)
		}
		vecs = append(vecs, tagSem[t])
	}
	sem = sum(vecs)
	if len(sem) == 0 {
		log.Printf("Semantics of state is undefined.")
	}
	return sem
}

func Initialize() {
	labelIds := make([]int, 0)
	lts := make(map[int][]string)
	tagDatasets = make(map[string][]string)
	labels = make(map[string]bool)
	labelsList = make([]string, 0)
	tagDomains = make(map[string][]string)
	tagNameSem = make(map[string][]float64)
	err := loadJson(GoodLabelsFile, &labelIds)
	if err != nil {
		panic(err)
	}
	labelIds = labelIds //[:100]
	labelNames = make(map[string]string)
	err = loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	// load label table
	err = loadJson(LabelTablesFile, &lts)
	if err != nil {
		panic(err)
	}
	for _, gl := range labelIds {
		labels[labelNames[strconv.Itoa(gl)]] = true
		tagDatasets[labelNames[strconv.Itoa(gl)]] = lts[gl]
		//tagDatasets[labelNames[strconv.Itoa(gl)]] = append(tagDatasets[labelNames[strconv.Itoa(gl)]], "testdataset")
	}
	// reading all domain embeddings
	domainSEmbs := make([][]string, 0)
	err = loadJson(DomainEmbsFile, &domainSEmbs)
	if err != nil {
		panic(err)
	}
	domainEmbs = make([][]float64, 0)
	domainEmbs = stringSlideToFloat(domainSEmbs)
	// reading table to emb id map
	tableEmbsMap = make(map[string][]int)
	err = loadJson(TableEmbsMap, &tableEmbsMap)
	if err != nil {
		panic(err)
	}
	// create domains
	for t, eids := range tableEmbsMap {
		for _, id := range eids {
			domains = append(domains, t+"_"+strconv.Itoa(id))
		}
	}
	// mapping tag to domains
	for _, gl := range labelIds {
		tagDomains[labelNames[strconv.Itoa(gl)]] = make([]string, 0)
		for _, d := range lts[gl] {
			for _, e := range tableEmbsMap[d] {
				tagDomains[labelNames[strconv.Itoa(gl)]] = append(tagDomains[labelNames[strconv.Itoa(gl)]], d+"_"+strconv.Itoa(e))
			}
		}
	}
	// load the embedding of each label
	getTagDomainEmbeddings()
	getTagNameEmbeddings()
	log.Println("len(tagNameSem): %d", len(tagNameSem))
	// eliminate labels without embeddings
	labels = make(map[string]bool)
	tm := make(map[string]bool)
	for _, gl := range labelIds {
		if _, ok := labelDomainEmbs[labelNames[strconv.Itoa(gl)]]; ok {
			labelsList = append(labelsList, labelNames[strconv.Itoa(gl)])
			tagDatasets[labelNames[strconv.Itoa(gl)]] = lts[gl]
			//tagDatasets[labelNames[strconv.Itoa(gl)]] = append(tagDatasets[labelNames[strconv.Itoa(gl)]], "testdataset")
			labels[labelNames[strconv.Itoa(gl)]] = true
			// adding tables of this label
			for _, t := range lts[gl] {
				if _, ok := tm[t]; !ok {
					tm[t] = true
				}
			}
		}
	}
	// making a list of all tables
	for t, _ := range tm {
		tables = append(tables, t)
	}
}

func getTagDomainEmbeddings() {
	dim := len(domainEmbs[0])
	labelDomainEmbs = make(map[string][][]float64)
	tagSem = make(map[string][]float64)
	for l, _ := range labels {
		lde := make([][]float64, 0)
		for _, t := range tagDatasets[l] {
			embIds := tableEmbsMap[t]
			for _, i := range embIds {
				// the first entry of an embedding slice is 0 and should be removed.
				lde = append(lde, domainEmbs[i][1:dim])
			}
		}
		if len(lde) == 0 {
			log.Printf("no emb for label %s", l)
			continue
		}
		labelDomainEmbs[l] = lde
		tagSem[l] = avg(lde)
	}
}

func getTagNameEmbeddings() {
	ft, err := InitInMemoryFastText(FasttextDb, func(v string) []string {
		stopWords := []string{"ckan_topiccategory_", "ckan_keywords_", "ckan_tags_", "ckan_subject_", "socrata_domaincategory_", "socrata_domaintags_", "socrata_tags_"}
		for _, st := range stopWords {
			v = strings.Replace(v, st, "", -1)
		}
		v = strings.Replace(strings.Replace(v, "_", " ", -1), "-", " ", -1)
		return strings.Split(v, " ")
	}, func(v string) string {
		return strings.ToLower(strings.TrimFunc(strings.TrimSpace(v), unicode.IsPunct))
	})
	if err != nil {
		panic(err)
	}
	for label, _ := range labels {
		embVec, err := ft.GetPhraseEmbMean(label)
		if err != nil {
			fmt.Printf("Error in building embedding for label %s : %s\n", label, err.Error())
			continue
		}
		tagNameSem[label] = embVec
	}
	return
}

func (org organization) Print() {
	for id, s := range org.states {
		fmt.Printf("%s: %v\n", id, s.tags)
	}
	log.Printf("transitions: %v", org.transitions)
	log.Printf("starts: %v", org.starts)
}
