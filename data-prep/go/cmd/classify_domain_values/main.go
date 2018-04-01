package main

import (
	"fmt"

	. "github.com/RJMillerLab/opendata-organization/opendata-ontology/data-prep/go/opendata"
)

func main() {
	CheckEnv()

	// Get the stream of filenames as a channel of strings
	filenames := StreamFilenames()

	// Classify the domains
	progress := DoClassifyDomainsFromFiles(10, filenames)

	start := GetNow()
	tick := start
	total := 0
	for n := range progress {
		total += n
		now := GetNow()

		if now-tick > 10 {
			tick = now
			fmt.Printf("Classified %d data files in %.2f seconds\n", total, GetNow()-start)
		}
	}
}
