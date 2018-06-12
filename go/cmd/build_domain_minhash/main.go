package main

import (
	"fmt"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/opendata"
)

func main() {
	CheckEnv()
	start := GetNow()
	filenames := StreamFilenames()
	sketches := DoMinhashNumericDomainsFromFiles(10, filenames, "values")
	progress := DoSaveDomainSketches(10, sketches, "minhash")
	total := ProgressCounter{}
	for n := range progress {
		total.Values += n.Values
		now := GetNow()
		if total.Values%100 == 0 {
			fmt.Printf("Processed %d domains in %.2f seconds\n", total.Values, now-start)
		}
	}
	fmt.Printf("Done generating minhash sketches for COD.")
}
