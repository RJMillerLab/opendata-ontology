package main

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"sync"
	"unicode"
	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/opendata"
)

func main() {
	CheckEnv()

	start := GetNow()
	ft, err := InitInMemoryFastText("/home/fnargesian/FASTTEXT/fasttext.db", func(v string) []string {
		return strings.Split(v, " ")
	}, func(v string) string {
		return strings.ToLower(strings.TrimFunc(strings.TrimSpace(v), unicode.IsPunct))
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("fasttext.db loaded in %.2f seconds.\n", GetNow()-start)

	filenames := StreamFilenames()
	valuefreqs := StreamValueFreqFromCache(10, filenames)

	progress := make(chan ProgressCounter)
	fanout := 40
	wg := &sync.WaitGroup{}
	wg.Add(fanout)
	for i := 0; i < fanout; i++ {
		go func() {
			for vf := range valuefreqs {
				// calculating mean
				log.Printf("file: %s - %d", vf.Filename, vf.Index)
				mean, size, err := ft.GetDomainEmbMean(vf.Values, vf.Freq)
				if err != nil {
					log.Printf("Error in building embedding for %s - %d: %s\n", vf.Filename, vf.Index, err.Error())
					continue
				}
				if size == 0 {
					log.Printf("No embedding representation found for %s.%d.", vf.Filename, vf.Index)
					continue
				}
				vecFilename := filepath.Join(OutputDir, "domains", fmt.Sprintf("%s/%d.ft-mean", vf.Filename, vf.Index))
				if err := WriteEmbVecToDisk(mean, vecFilename); err != nil {
					panic(err)
				}
			}
			wg.Done()
		}()
	}
	go func() {
		wg.Wait()
		close(progress)
	}()
	i := 0
	total := ProgressCounter{}
	for n := range progress {
		total.Values += n.Values
		i += 1
		log.Printf("Processed %d domains.", i, total.Values)
	}
	log.Printf("Finished counting %d domains.", total.Values)
}
