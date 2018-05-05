package main

import (
	"log"
	"sync"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/ontology"
)

func main() {
	// Stream all possible organizations
	organizations := make([][][]int, 0)
	overlaps := make(map[int]map[int]int)
	wg := &sync.WaitGroup{}
	wg.Add(2)
	go func() {
		organizations = Generate2DimOrganizations()
		log.Println("finished generating orgs.")
		wg.Done()
	}()
	go func() {
		// Compute all possible overlaps
		overlaps = ComputeLabelOverlaps()
		log.Println("finished overlap calc.")
		wg.Done()
	}()
	wg.Wait()
	// Find the best organization
	bestOrg, bestScore := FindOrganization(organizations, overlaps)
	log.Printf("best org: %v score: %f", bestOrg, bestScore)
	PrintOrganization(bestOrg, overlaps)
}
