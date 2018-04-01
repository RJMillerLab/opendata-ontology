package main

import (
	"fmt"
	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/opendata"
)

// Use 30 threads for reading the filename stream
const nReaders = 30
const nWriters = 30

func main() {
	CheckEnv()

	// Get the channel of filenames (as string)
	filenames := StreamFilenames()

	// Map filenames to domain fragments using
	// nReader goroutines
	domains := StreamDomainsFromFilenames(nReaders, filenames)

	// Save the domain fragments to disk
	// and report the progress with the progress
	// counter channel
	progress := DoSaveDomainValues(nWriters, domains)

	i := 0
	total := ProgressCounter{}
	start := GetNow()
	tick := GetNow()
	for n := range progress {
		total.Values += n.Values
		i += 1
		now := GetNow()

		if now-tick > 10 {
			tick = now
			fmt.Printf("[fragment %d] written %d values in %.2f seconds\n", i, total.Values, now-start)
		}
	}

	fmt.Printf("Done, written %d values in %.2f seconds\n", total.Values, GetNow()-start)
}
