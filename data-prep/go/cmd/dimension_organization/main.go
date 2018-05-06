package main

import (
	"log"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/ontology"
)

func main() {
	// Compute all possible overlaps
	overlaps := ComputeLabelOverlaps()
	log.Println("finished overlap calc.")
	organizations := Generate2DimOrganizations()
	// Find the best organization
	bestOrgs := FindOrganization(organizations, overlaps)
	PrintOrganization(bestOrgs, overlaps)
	log.Printf("Done!")
}
