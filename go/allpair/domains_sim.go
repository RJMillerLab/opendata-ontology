package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"math"

	"github.com/RJMillerLab/table-union/simhashlsh"
)

func main() {
	domains := make(map[string][]float64)
	sims := make(map[string]map[string]float64)
	//err := loadJson("/home/fnargesian/FINDOPENDATA_DATASETS/socrata/socrata_domain_40051_map_embs.json", &domains)
	err := loadJson("/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/nometa_output/nometa_domains.json", &domains)
	if err != nil {
		panic(err)
	}
	clsh := simhashlsh.NewCosineLSH(300, 100, 0.5)
	count := 0
	for name, dom := range domains {
		count += 1
		if count%200 == 0 {
			log.Printf("Added %d domains to index.", count)
		}
		clsh.Add(dom, name)
	}
	log.Printf("started indexing")
	clsh.Index()
	log.Printf("done indexing")
	//
	seenpairs := make(map[string]bool)
	count = 0
	for name, dom := range domains {
		count += 1
		if count%200 == 0 {
			log.Printf("queried %d domains.", count)
			log.Printf("%d in map", len(sims))
		}
		results := clsh.Query(dom)
		for _, r := range results {
			if _, ok := seenpairs[r+name]; !ok {
				if _, ok := seenpairs[name+r]; !ok {
					s := cosine(dom, domains[r])
					if s < 0.5 {
						continue
					}
					if _, ok := sims[name]; !ok {
						sims[name] = make(map[string]float64)
					}
					if _, ok := sims[r]; !ok {
						sims[r] = make(map[string]float64)
					}
					sims[r][name] = s
					sims[name][r] = s
				}
			}
		}
	}
	log.Println(len(sims))
	dumpJson("/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/python/nometa_output/allpair_sims.json", sims)
}

func loadJson(file string, v interface{}) (err error) {
	buffer, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	err = json.Unmarshal(buffer, v)
	if err != nil {
		return err
	}
	return nil
}

func cosine(x, y []float64) float64 {
	if len(x) != len(y) {
		log.Printf("%d vs %d", len(x), len(y))
		panic("Length of vectors not equal")
	}
	dot := 0.0
	modX, modY := 0.0, 0.0
	for i := range x {
		dot += x[i] * y[i]
		modX += x[i] * x[i]
		modY += y[i] * y[i]
	}
	return dot / (math.Sqrt(modX) * math.Sqrt(modY))
}

func dumpJson(file string, v interface{}) error {
	buffer, err := json.Marshal(v)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(file, buffer, 0664)
	if err != nil {
		return err
	}
	return nil
}
