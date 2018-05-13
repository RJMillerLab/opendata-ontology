package main

import (
	"fmt"
	"log"

	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/ontology"
)

func main() {
	// Compute all possible overlaps
	overlaps := ComputeLabelOverlaps()
	log.Println("finished overlap calc.")
	organizations := Generate2DimOrganizations()
	// Find the best organization
	bestOrgsDensity, bestOrgsUniformity, bestOrgsAgg := FindOrganization(organizations, overlaps)
	fmt.Printf("density:")
	PrintOrganization(bestOrgsDensity, overlaps)
	fmt.Printf("uniformity:")
	PrintOrganization(bestOrgsUniformity, overlaps)
	fmt.Printf("aggregate:")
	PrintOrganization(bestOrgsAgg, overlaps)
	log.Printf("Done!")
}
