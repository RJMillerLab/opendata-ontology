package main

import (
	"log"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/ontology"
)

func main() {
	// Stream all possible organizations
	organizations := ReadOrganzations()
	os := Generate2DimOrganizations(4)
	log.Println(os)
	// Compute all possible overlaps
	overlaps := ComputeLabelOverlaps()
	// Find the best organization
	bestOrg := FindBestOrganization(organizations, overlaps)
	log.Printf("best org: %v", bestOrg)
}
