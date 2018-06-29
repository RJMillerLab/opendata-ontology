package hierarchy

import (
	"fmt"
	"log"
	"strconv"

	"github.com/RJMillerLab/opendata-ontology/data-prep/go/pqueue"
)

var (
	tags          map[string]bool
	tables        []string
	tagNames      map[string]string
	tagsList      []string
	tagEmbs       map[string][]float64
	tagDomainEmbs map[string][][]float64
	tagAvgEmb     map[string][]float64
	tableEmbsMap  map[string][]int
	tagDomains    map[string][]string
	domainEmbs    [][]float64
)

type pair struct {
	cId1 int
	cId2 int
	c1   cluster
	c2   cluster
}

type cluster struct {
	tags       []string
	sem        []float64
	population [][]float64
	id         int
}

type clustering struct {
	clusters        []cluster
	distances       [][]float64 // distance matrix
	hierarchy       map[int][]int
	closestClusters []int
	mergeQueue      *pqueue.TopKQueue
	mergedClusters  map[int]bool // index of clusters in clusters that have been already merged.
	singleClusters  map[int]bool
	threshold       float64
	root            cluster
}

func newCluster(tagname string, id int) cluster {
	tags := make([]string, 0)
	tags = append(tags, tagname)
	return cluster{
		tags:       tags,
		population: tagDomainEmbs[tagname],
		sem:        tagAvgEmb[tagname],
		id:         id,
	}
}

func newClustering() *clustering {
	clusters := initializeClusters()
	log.Printf("initiliazation: ")
	cing := &clustering{
		clusters:       clusters,
		mergeQueue:     pqueue.NewTopKQueue(len(clusters)),
		mergedClusters: make(map[int]bool),
		singleClusters: make(map[int]bool),
		threshold:      0.85,
		hierarchy:      make(map[int][]int),
	}
	cing.initializeDistanceMatrix()
	return cing
}

func initializeClusters() []cluster {
	clusters := make([]cluster, 0)
	for _, l := range tagsList {
		clusters = append(clusters, newCluster(l, len(clusters)))
	}
	return clusters
}

func (cing *clustering) initializeDistanceMatrix() {
	cing.distances = make([][]float64, len(cing.clusters))
	// initiliazing the matrix
	for i := 0; i < len(cing.clusters); i++ {
		cing.distances[i] = make([]float64, len(cing.clusters))
	}
	//mind := 2.0
	for i := 0; i < len(cing.clusters); i++ {
		cing.singleClusters[i] = true
		for j := i + 1; j < len(cing.clusters); j++ {
			d := Cosine(cing.clusters[i].sem, cing.clusters[j].sem)
			//p := pair{c1: i, c2: j}
			p := pair{c1: cing.clusters[i], cId1: cing.clusters[i].id, c2: cing.clusters[j], cId2: cing.clusters[j].id}
			if d > 0.0 {
				cing.mergeQueue.Push(p, d)
			} else {
				log.Printf("zero dist in merge")
			}
			cing.distances[i][j] = d
			cing.distances[j][i] = d
		}
	}
	//log.Printf("init cing.mergeQueue.Size(): %d", cing.mergeQueue.Size())
}

func (cing *clustering) MakeThresholdClustering() {
	c, init := cing.findClustersToMergeThreshold()
	for cond := init; cond; {
		cing.addNewCluster(c)
		c, cond = cing.findClustersToMergeThreshold()
	}
	log.Printf("done clustering.")
	cing.printClustring()
}

func (cing *clustering) BuildClusters() {
	p, init := cing.findClustersToMerge()
	for cond := init; cond; { //cond = cing.doneClustering() {
		//p, cond := cing.findClustersToMerge()
		cing.mergeClusters(p)
		p, cond = cing.findClustersToMerge()
	}
	log.Printf("done clustering.")
	cing.printClustring()
}

func (cing *clustering) findClustersToMergeThreshold() (cluster, bool) {
	mp, s := cing.mergeQueue.Pop()
	merge := cluster{}
	found := false
	for e := mp; !found && e != nil; e = mp {
		p := e.(pair)
		if _, ok1 := cing.mergedClusters[p.cId1]; !ok1 {
			if _, ok2 := cing.mergedClusters[p.cId2]; !ok2 {
				found = true
				log.Printf("found the seed: %d %d: %f", p.cId1, p.cId2, s)
				merge = cing.buildClusterFromSeed(p)
			}
		}
		if !found {
			mp, s = cing.mergeQueue.Pop()
		}
	}
	if found == true {
		return merge, true
	} else {
		return cluster{}, false
	}
}

func (cing *clustering) addNewCluster(nc cluster) {
	//log.Printf("adding new cluster.")
	for i, c := range cing.clusters {
		if _, ok := cing.mergedClusters[i]; !ok {
			d := Cosine(c.sem, nc.sem)
			p := pair{cId1: cing.clusters[i].id, cId2: nc.id, c1: cing.clusters[i], c2: nc}
			if d > 0.0 {
				cing.mergeQueue.Push(p, d)
			} else {
				log.Printf("zero dist in merge")
			}
		}
	}
	cing.clusters = append(cing.clusters, nc)
	//log.Printf("number of clusters now: %d", len(cing.clusters))
}

func (cing *clustering) buildClusterFromSeed(p pair) cluster {
	delete(cing.singleClusters, p.cId1)
	delete(cing.singleClusters, p.cId2)
	cing.mergedClusters[p.cId1] = true
	cing.mergedClusters[p.cId2] = true
	seed := cluster{
		tags:       append(p.c1.tags, p.c2.tags...),
		population: append(p.c1.population, p.c2.population...),
		sem:        mergeClusterSem(p.c1.sem, p.c2.sem),
		id:         len(cing.clusters),
	}
	mcount := 2
	if len(cing.singleClusters) == 0 {
		//log.Println("len(cing.singleClusters)=0")
		return seed
	}
	aboveThreshold := false
	for !aboveThreshold {
		seedQueue := pqueue.NewTopKQueue(len(cing.singleClusters))
		for ci, _ := range cing.singleClusters {
			d := Cosine(cing.clusters[ci].sem, seed.sem)
			if d > 0.0 {
				p := pair{c1: cing.clusters[ci], c2: seed, cId1: ci}
				seedQueue.Push(p, d)
			}
		}
		if seedQueue.Size() == 0 {
			aboveThreshold = true
			continue
		}
		p, _ := seedQueue.Pop()
		mp := p.(pair)
		//log.Printf("should i merge %d with score %f with the seed?", mp.cId1, s)
		merge := cluster{
			tags:       append(mp.c1.tags, mp.c2.tags...),
			population: append(mp.c1.population, mp.c2.population...),
			sem:        mergeClusterSem(mp.c1.sem, mp.c2.sem),
			id:         len(cing.clusters),
		}
		d := Cosine(merge.sem, seed.sem)
		if d < cing.threshold {
			//log.Println("no")
			aboveThreshold = true
		} else {
			mcount += 1
			seed = merge
			cing.mergedClusters[mp.cId1] = true
			delete(cing.singleClusters, mp.cId1)
		}
	}
	log.Printf("merged %d clusters.", mcount)
	return seed
}

func (cing *clustering) findClustersToMerge() (pair, bool) {
	found := false
	mp, _ := cing.mergeQueue.Pop()
	//log.Printf("start mp: %v ms: %f", mp, ms)
	for e := mp; !found && e != nil; e = mp {
		//log.Printf("loop mp: %v ms: %f", mp, ms)
		p := e.(pair)
		if _, ok1 := cing.mergedClusters[p.cId1]; !ok1 {
			if _, ok2 := cing.mergedClusters[p.cId2]; !ok2 {
				found = true
			}
		}
		if !found {
			mp, _ = cing.mergeQueue.Pop()
		}
		//log.Printf("found: %v", found)
	}
	if found == true {
		//log.Printf("merging %d and %d with distance %f", mp.(pair).c1, mp.(pair).c2, ms)
		return mp.(pair), true
	} else {
		return pair{cId1: -1, cId2: -1}, false
	}
}

func (cing *clustering) mergeClusters(clusterPair pair) {
	cinx1 := clusterPair.cId1
	cinx2 := clusterPair.cId2
	c1 := cing.clusters[cinx1]
	c2 := cing.clusters[cinx2]
	nc := cluster{
		tags:       append(c1.tags, c2.tags...),
		population: append(c1.population, c2.population...),
		sem:        mergeClusterSem(c1.sem, c2.sem),
		id:         len(cing.clusters),
	}
	if len(cing.clusters) == 0 {
		cing.root = nc
	}
	for i, c := range cing.clusters {
		if _, ok := cing.mergedClusters[i]; !ok {
			d := Cosine(c.sem, nc.sem)
			// the index of the new cluster will be the current size of the clusters
			p := pair{cId1: i, cId2: len(cing.clusters)}
			if d > 0.0 {
				cing.mergeQueue.Push(p, d)
			} else {
				log.Printf("zero dist in merge")
			}
		}
	}
	cing.clusters = append(cing.clusters, nc)
	cing.mergedClusters[cinx1] = true
	cing.mergedClusters[cinx2] = true
	if _, ok := cing.hierarchy[nc.id]; !ok {
		cing.hierarchy[nc.id] = make([]int, 0)
	}
	cing.hierarchy[nc.id] = append(cing.hierarchy[nc.id], cinx1)
	cing.hierarchy[nc.id] = append(cing.hierarchy[nc.id], cinx2)
}

func (cing *clustering) mergeClustersOld(clusterPair pair) {
	// create and add a new cluster with a new sem vector
	cinx1 := clusterPair.cId1
	cinx2 := clusterPair.cId2
	c1 := cing.clusters[cinx1]
	c2 := cing.clusters[cinx2]
	nc := cluster{
		tags:       append(c1.tags, c2.tags...),
		population: append(c1.population, c2.population...),
		sem:        mergeClusterSem(c1.sem, c2.sem),
		id:         len(cing.clusters),
	}
	// remove the clusters from the distance matrix
	// TODO: for now, just copying the matrix to a new one
	// and skipping the row and column related to the clusters
	nm := make([][]float64, 0)
	for i := 0; i < len(cing.distances); i++ {
		if i != cinx1 && i != cinx2 {
			row := make([]float64, 0)
			for j := 0; j < len(cing.distances[i]); j++ {
				if j != cinx1 && j != cinx2 {
					row = append(row, cing.distances[i][j])
				}
			}
			nm = append(nm, row)
		}
	}
	cing.distances = nm
	// removing merged clusters and adding the new one
	cs := make([]cluster, 0)
	for i, c := range cing.clusters {
		if i != cinx1 && i != cinx2 {
			cs = append(cs, c)
		}
	}
	cing.clusters = cs
	// update distances
	// compue the distance of the new cluster with all clusters
	nrow := make([]float64, 0)
	for i, c := range cing.clusters {
		d := Cosine(c.sem, nc.sem)
		nrow = append(nrow, d)
		// the index of the new cluster will be the current size of the clusters
		p := pair{cId1: i, cId2: len(cing.clusters)}
		if d > 0.0 {
			cing.mergeQueue.Push(p, d)
		} else {
			log.Printf("zero dist in merge")
		}
		nrow = append(nrow, d)
		cing.distances[i] = append(cing.distances[i], d)
	}
	cing.clusters = append(cing.clusters, nc)
	cing.distances = append(cing.distances, nrow)
}

func Initialize() *clustering {
	// load tags
	tagIds := make([]int, 0)
	lts := make(map[int][]string)
	tagDomains = make(map[string][]string)
	tags = make(map[string]bool)
	tagEmbs = make(map[string][]float64)
	tagsList = make([]string, 0)
	err := loadJson(GoodTagsFile, &tagIds)
	if err != nil {
		panic(err)
	}
	tagIds = tagIds[:50]
	tagNames = make(map[string]string)
	err = loadJson(TagNamesFile, &tagNames)
	if err != nil {
		panic(err)
	}
	// load tag table
	err = loadJson(TagTablesFile, &lts)
	if err != nil {
		panic(err)
	}
	for _, gl := range tagIds {
		tags[tagNames[strconv.Itoa(gl)]] = true
		tagDomains[tagNames[strconv.Itoa(gl)]] = lts[gl]
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
	// load the embedding of each tag
	getTagDomainEmbeddings()
	// eliminate tags without embeddings
	tm := make(map[string]bool)
	for _, gl := range tagIds {
		if _, ok := tagDomainEmbs[tagNames[strconv.Itoa(gl)]]; ok {
			tagsList = append(tagsList, tagNames[strconv.Itoa(gl)])
			tagDomains[tagNames[strconv.Itoa(gl)]] = lts[gl]
			tags[tagNames[strconv.Itoa(gl)]] = true
			// adding tables of this tag
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
	return newClustering()
}

func getTagDomainEmbeddings() (map[string][][]float64, map[string][]float64) {
	dim := len(domainEmbs[0])
	tagDomainEmbs = make(map[string][][]float64)
	tagAvgEmb = make(map[string][]float64)
	for l, _ := range tags {
		lde := make([][]float64, 0)
		for _, t := range tagDomains[l] {
			embIds := tableEmbsMap[t]
			for _, i := range embIds {
				// the first entry of an embedding slice is 0 and should be removed.
				lde = append(lde, domainEmbs[i][1:dim])
			}
		}
		tagDomainEmbs[l] = lde
		tagAvgEmb[l] = avg(lde)
	}
	return tagDomainEmbs, tagAvgEmb
}

func (cing *clustering) doneClustering() bool {
	if cing.mergeQueue.Size() == 0 {
		//if len(cing.distances) == 1 {
		return true
	}
	return false
}

func mergeClusterSem(s1, s2 []float64) []float64 {
	a := make([]float64, 0)
	for i, v1 := range s1 {
		a = append(a, (v1+s2[i])/2.0)
	}
	return a
}

func (cing *clustering) printClustring() {
	for i, c := range cing.clusters {
		if len(c.tags) > 1 {
			fmt.Printf("cluster %d: %v\n", i, c.tags)
		}
	}
	for s, ts := range cing.hierarchy {
		fmt.Printf("%d -> (%v)", s, ts)
	}
}
