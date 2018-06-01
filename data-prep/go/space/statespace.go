package space

import (
	"fmt"
	"log"
	"strconv"

	"github.com/RJMillerLab/opendata-ontology/data-prep/go/pqueue"
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
)

//type space struct {
//	states      []node
//	transitions map[int][]int
//}

type pair struct {
	c1 int
	c2 int
}

type cluster struct {
	labels     []string
	sem        []float64
	population [][]float64
}

type clustering struct {
	clusters        []cluster
	distances       [][]float64 // distance matrix
	hierarchy       map[int][]int
	closestClusters []int
	mergeQueue      *pqueue.TopKQueue
	mergedClusters  map[int]bool // index of clusters in clusters that have been already merged.
}

func newCluster(labelname string) cluster {
	labels := make([]string, 0)
	labels = append(labels, labelname)
	return cluster{
		labels:     labels,
		population: labelDomainEmbs[labelname],
		sem:        labelAvgEmb[labelname],
	}
}

func newClustering() *clustering {
	clusters := initializeClusters()
	log.Printf("initiliazation: ")
	cing := &clustering{
		clusters:       clusters,
		mergeQueue:     pqueue.NewTopKQueue(len(clusters)),
		mergedClusters: make(map[int]bool),
	}
	cing.initializeDistanceMatrix()
	return cing
}

func initializeClusters() []cluster {
	clusters := make([]cluster, 0)
	for _, l := range labelsList {
		clusters = append(clusters, newCluster(l))
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
		for j := i + 1; j < len(cing.clusters); j++ {
			d := Cosine(cing.clusters[i].sem, cing.clusters[j].sem)
			p := pair{c1: i, c2: j}
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

func (cing *clustering) BuildClusters() {
	p, init := cing.findClustersToMerge()
	for cond := init; cond; { //cond = cing.doneClustering() {
		//p, cond := cing.findClustersToMerge()
		//log.Printf("# clusters before merge: %d", len(cing.clusters))
		cing.mergeClusters(p)
		//log.Printf("# clusters after merge: %d", len(cing.clusters))
		p, cond = cing.findClustersToMerge()
	}
	log.Printf("done clustering.")
	cing.printClustring()
}

func (cing *clustering) findClustersToMerge() (pair, bool) {
	found := false
	mp, _ := cing.mergeQueue.Pop()
	//log.Printf("start mp: %v ms: %f", mp, ms)
	for e := mp; !found && e != nil; e = mp {
		//log.Printf("loop mp: %v ms: %f", mp, ms)
		p := e.(pair)
		if _, ok1 := cing.mergedClusters[p.c1]; !ok1 {
			if _, ok2 := cing.mergedClusters[p.c2]; !ok2 {
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
		return pair{c1: -1, c2: -1}, false
	}
}

func (cing *clustering) mergeClusters(clusterPair pair) {
	//log.Printf("starting merge by cing.mergeQueue.Size(): %d", cing.mergeQueue.Size())
	cinx1 := clusterPair.c1
	cinx2 := clusterPair.c2
	c1 := cing.clusters[cinx1]
	c2 := cing.clusters[cinx2]
	nc := cluster{
		labels:     append(c1.labels, c2.labels...),
		population: append(c1.population, c2.population...),
		sem:        mergeClusterSem(c1.sem, c2.sem),
	}
	for i, c := range cing.clusters {
		if _, ok := cing.mergedClusters[i]; !ok {
			d := Cosine(c.sem, nc.sem)
			// the index of the new cluster will be the current size of the clusters
			p := pair{c1: i, c2: len(cing.clusters)}
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
}

func (cing *clustering) mergeClustersOld(clusterPair pair) {
	log.Printf("starting merge by cing.mergeQueue.Size(): %d", cing.mergeQueue.Size())
	// create and add a new cluster with a new sem vector
	cinx1 := clusterPair.c1
	cinx2 := clusterPair.c2
	c1 := cing.clusters[cinx1]
	c2 := cing.clusters[cinx2]
	nc := cluster{
		labels:     append(c1.labels, c2.labels...),
		population: append(c1.population, c2.population...),
		sem:        mergeClusterSem(c1.sem, c2.sem),
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
		p := pair{c1: i, c2: len(cing.clusters)}
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
	log.Printf("len(cing.clusters): %d", len(cing.clusters))
	log.Printf("len(cing.distances): %d", len(cing.distances))
	log.Printf("cing.mergeQueue.Size(): %d", cing.mergeQueue.Size())
}

func Initialize() *clustering {
	// load labels
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
	labelIds = labelIds //[:10]
	log.Printf("len(labelIds): %d", len(labelIds))
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
	// load the embedding of each label
	getLabelDomainEmbeddings()
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
	return newClustering()
}

func getLabelDomainEmbeddings() (map[string][][]float64, map[string][]float64) {
	dim := len(domainEmbs[0])
	labelDomainEmbs = make(map[string][][]float64)
	labelAvgEmb = make(map[string][]float64)
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
		labelAvgEmb[l] = avg(lde)
	}
	return labelDomainEmbs, labelAvgEmb
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
		if len(c.labels) > 1 {
			fmt.Printf("cluster %d: %v\n", i, c.labels)
		}
	}
}
