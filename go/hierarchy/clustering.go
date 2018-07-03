package hierarchy

import (
	"fmt"
	"log"
	"strconv"

	"github.com/RJMillerLab/opendata-ontology/data-prep/go/pqueue"
)

var (
	tags         map[string]bool
	tables       []string
	tagNames     map[string]string
	tagsList     []string
	tagSems      map[string][]float64
	tableEmbsMap map[string][]int
	domainEmbs   [][]float64

	//
	domainSems    map[string][]float64
	domainTags    map[string]string
	tableTags     map[string][]string
	tagDomains    map[string][]string
	tagTables     map[string][]string
	tagDomainSems map[string][][]float64
)

type pair struct {
	cId1 int
	cId2 int
	//c1   cluster
	//c2   cluster
}

type cluster struct {
	tags       []string
	sem        []float64
	population [][]float64
	id         int
	parents    []int
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
		population: tagDomainSems[tagname],
		sem:        tagSems[tagname],
		id:         id,
		parents:    make([]int, 0),
	}
}

func newClustering() *clustering {
	clusters := initializeClusters()
	cing := &clustering{
		clusters:       clusters,
		mergeQueue:     pqueue.NewTopKQueue(3 * len(clusters) * len(clusters)),
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
	//for _, l := range tagsList {
	for l, _ := range tagSems {
		clusters = append(clusters, newCluster(l, len(clusters)))
	}
	//log.Printf("len(clusters): %d", len(clusters))
	return clusters
}

func (cing *clustering) initializeDistanceMatrix() {
	cing.distances = make([][]float64, len(cing.clusters))
	// initiliazing the matrix
	for i := 0; i < len(cing.clusters); i++ {
		cing.distances[i] = make([]float64, len(cing.clusters))
	}
	//mind := 2.0
	max := -1.0
	min := 1.0
	for i := 0; i < len(cing.clusters); i++ {
		//cing.singleClusters[i] = true
		for j := i + 1; j < len(cing.clusters); j++ {
			d := cosine(cing.clusters[i].sem, cing.clusters[j].sem)
			p := pair{cId1: cing.clusters[i].id, cId2: cing.clusters[j].id}
			//p := pair{c1: cing.clusters[i], cId1: cing.clusters[i].id, c2: cing.clusters[j], cId2: cing.clusters[j].id}
			if d > 0.0 {
				if d > max {
					max = d
				}
				if d < min {
					min = d
				}
				cing.mergeQueue.Push(p, -1.0*d)
			} else {
				log.Printf("zero dist in merge")
			}
			cing.distances[i][j] = d
			cing.distances[j][i] = d
		}
	}
}

func (cing *clustering) BuildClusters() {
	p, init := cing.findClustersToMerge()
	for cond := init; cond; { //cond = cing.doneClustering() {
		//p, cond := cing.findClustersToMerge()
		cing.mergeClusters(p)
		p, cond = cing.findClustersToMerge()
	}
	if len(cing.clusters) != 0 {
		cing.root = cing.clusters[len(cing.clusters)-1]
		log.Printf("root: %d", cing.root.id)
	}

	//log.Printf("done clustering.")
	//cing.printClustring()
}

func (cing *clustering) MakeThresholdClustering() {
	c, init := cing.findAndMergeClustersThreshold()
	for cond := init; cond; {
		cing.addNewCluster(c)
		c, cond = cing.findAndMergeClustersThreshold()
	}
	if len(cing.clusters) != 0 {
		cing.root = cing.clusters[len(cing.clusters)-1]
		log.Printf("root: %d", cing.root.id)
	}
	log.Printf("done clustering.")
	cing.printClustring()
}

func (cing *clustering) findAndMergeClustersThreshold() (cluster, bool) {
	mp, _ := cing.mergeQueue.Pop()
	merge := cluster{}
	found := false
	for e := mp; !found && e != nil; e = mp {
		p := e.(pair)
		if _, ok1 := cing.mergedClusters[p.cId1]; !ok1 {
			if _, ok2 := cing.mergedClusters[p.cId2]; !ok2 {
				found = true
				merge = cing.buildClusterFromSeed(p)
			}
		}
		if !found {
			mp, _ = cing.mergeQueue.Pop()
		}
	}
	if found == true {
		return merge, true
	} else {
		return cluster{}, false
	}
}

func (cing *clustering) addNewCluster(nc cluster) {
	for i, c := range cing.clusters {
		if c.id == nc.id {
			continue
		}
		if _, ok := cing.mergedClusters[i]; !ok {
			d := cosine(c.sem, nc.sem)
			p := pair{cId1: c.id, cId2: nc.id}
			if d > 0.0 {
				cing.mergeQueue.Push(p, -1.0*d)
			} else {
				log.Printf("zero dist in merge")
			}
		}
	}
	cing.clusters = append(cing.clusters, nc)
}

func (cing *clustering) buildClusterFromSeed(p pair) cluster {
	cing.mergedClusters[p.cId1] = true
	cing.mergedClusters[p.cId2] = true
	p1 := cing.clusters[p.cId1]
	p2 := cing.clusters[p.cId2]
	seed := cluster{
		tags:       append(p1.tags, p2.tags...),
		population: append(p1.population, p2.population...),
		sem:        mergeClusterSem(p1.sem, p2.sem),
		id:         len(cing.clusters),
	}
	cing.clusters = append(cing.clusters, seed)
	if _, ok := cing.hierarchy[seed.id]; !ok {
		cing.hierarchy[seed.id] = make([]int, 0)
	}
	cing.hierarchy[seed.id] = append(cing.hierarchy[seed.id], p.cId1)
	cing.hierarchy[seed.id] = append(cing.hierarchy[seed.id], p.cId2)
	cing.clusters[p.cId1].parents = append(cing.clusters[p.cId1].parents, seed.id)
	cing.clusters[p.cId2].parents = append(cing.clusters[p.cId2].parents, seed.id)
	mcount := 2
	seedQueue := pqueue.NewTopKQueue(len(cing.clusters))
	for ci, _ := range cing.clusters {
		if cing.clusters[ci].id == seed.id {
			continue
		}
		if _, ok := cing.mergedClusters[cing.clusters[ci].id]; ok {
			continue
		}
		d := cosine(cing.clusters[ci].sem, seed.sem)
		if d > 0.0 {
			p := pair{cId1: cing.clusters[ci].id, cId2: seed.id}
			seedQueue.Push(p, -1.0*d)
		}
	}
	aboveThreshold := false
	for !aboveThreshold {
		if seedQueue.Size() == 0 {
			log.Printf("seedQueue.Size() == 0")
			aboveThreshold = true
			continue
		}
		p, _ := seedQueue.Pop()
		mp := p.(pair)
		mp1 := cing.clusters[mp.cId1]
		mp2 := cing.clusters[mp.cId2]
		merge := cluster{
			tags:       append(mp1.tags, mp2.tags...),
			population: append(mp1.population, mp2.population...),
			sem:        mergeClusterSem(mp1.sem, mp2.sem),
			id:         seed.id,
		}
		d := cosine(merge.sem, seed.sem)
		if d < cing.threshold {
			aboveThreshold = true
		} else {
			cing.clusters[seed.id] = merge
			mcount += 1
			seed = merge
			cing.mergedClusters[mp.cId1] = true
			cing.hierarchy[seed.id] = append(cing.hierarchy[seed.id], mp.cId1)
			cing.clusters[mp.cId1].parents = append(cing.clusters[mp.cId1].parents, seed.id)
		}
	}
	return seed
}

func (cing *clustering) findClustersToMerge() (pair, bool) {
	found := false
	mp, _ := cing.mergeQueue.Pop()
	for e := mp; !found && e != nil; e = mp {
		p := e.(pair)
		if _, ok1 := cing.mergedClusters[p.cId1]; !ok1 {
			if _, ok2 := cing.mergedClusters[p.cId2]; !ok2 {
				found = true
			}
		}
		if !found {
			mp, _ = cing.mergeQueue.Pop()
		}
	}
	if found == true {
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
	for i, c := range cing.clusters {
		if _, ok := cing.mergedClusters[i]; !ok {
			d := cosine(c.sem, nc.sem)
			// the index of the new cluster will be the current size of the clusters
			p := pair{cId1: i, cId2: len(cing.clusters)}
			if d > 0.0 {
				cing.mergeQueue.Push(p, -1.0*d)
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
	cing.clusters[cinx1].parents = append(cing.clusters[cinx1].parents, nc.id)
	cing.clusters[cinx2].parents = append(cing.clusters[cinx2].parents, nc.id)
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
		d := cosine(c.sem, nc.sem)
		nrow = append(nrow, d)
		// the index of the new cluster will be the current size of the clusters
		p := pair{cId1: i, cId2: len(cing.clusters)}
		if d > 0.0 {
			cing.mergeQueue.Push(p, -1.0*d)
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
	domainSems, domainTags, tableTags, tagDomains, tagTables, tagDomainSems, tagSems = buildContext()
	return newClustering()
}

func InitializePlus() *clustering {
	// load tags
	tagIds := make([]int, 0)
	lts := make(map[int][]string)
	tagDomains = make(map[string][]string)
	tags = make(map[string]bool)
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
		if _, ok := tagDomainSems[tagNames[strconv.Itoa(gl)]]; ok {
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
	tagDomainSems = make(map[string][][]float64)
	tagSems = make(map[string][]float64)
	for l, _ := range tags {
		lde := make([][]float64, 0)
		for _, t := range tagDomains[l] {
			embIds := tableEmbsMap[t]
			for _, i := range embIds {
				// the first entry of an embedding slice is 0 and should be removed.
				lde = append(lde, domainEmbs[i][1:dim])
			}
		}
		tagDomainSems[l] = lde
		tagSems[l] = avg(lde)
	}
	return tagDomainSems, tagSems
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
		fmt.Printf("%d -> (%v)\n", s, ts)
	}
}
