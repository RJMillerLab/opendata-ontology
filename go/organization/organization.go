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
)

var (
	labels          map[string]bool
	tables          []string
	labelNames      map[string]string
	labelsList      []string
	labelEmbs       map[string][]float64
	labelDomainEmbs map[string][][]float64
	labelAvgEmb     map[string][]float64
	tableEmbsMap    map[string][]int
	labelTables     map[string][]string
	domainEmbs      [][]float64

	tagSem        map[string][]float64
	tagDatasets   map[string]map[string][]string
	startStateNum = 5
	stateStopProb = 0.5
	numRuns       = 5
)

type state struct {
	tags []string
	sem  []float64
	name string
	id   string
}

type organization struct {
	states      map[string]state    // mapping from id to state
	transitions map[string][]string // state transitions
	starts      []string            // a list of state ids
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
		s := state{
			id:   id,
			tags: tags,
			sem:  sem,
		}
		fmt.Printf("state %s\n", id)
		states[id] = s
	}
	fmt.Printf("len(states): %d\n", len(states))
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
	fmt.Printf("transitions: %v\n", transitions)
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
	fmt.Printf("starts: %v\n", starts)
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
	labelTables = make(map[string][]string)
	labels = make(map[string]bool)
	labelEmbs = make(map[string][]float64)
	labelsList = make([]string, 0)
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
		labelTables[labelNames[strconv.Itoa(gl)]] = lts[gl]
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
	log.Println(labels)
	for l, _ := range labels {
		fmt.Printf("%s\n", l)
	}

	// load the embedding of each label
	getTagDomainEmbeddings()
	// eliminate labels without embeddings
	tm := make(map[string]bool)
	for _, gl := range labelIds {
		if _, ok := labelDomainEmbs[labelNames[strconv.Itoa(gl)]]; ok {
			labelsList = append(labelsList, labelNames[strconv.Itoa(gl)])
			labelTables[labelNames[strconv.Itoa(gl)]] = lts[gl]
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
	log.Printf("len(tagSem): %d", len(tagSem))
}

func getTagDomainEmbeddings() {
	dim := len(domainEmbs[0])
	labelDomainEmbs = make(map[string][][]float64)
	tagSem = make(map[string][]float64)
	for l, _ := range labels {
		lde := make([][]float64, 0)
		for _, t := range labelTables[l] {
			embIds := tableEmbsMap[t]
			for _, i := range embIds {
				// the first entry of an embedding slice is 0 and should be removed.
				lde = append(lde, domainEmbs[i][1:dim])
			}
		}
		labelDomainEmbs[l] = lde
		tagSem[l] = avg(lde)
	}
}
