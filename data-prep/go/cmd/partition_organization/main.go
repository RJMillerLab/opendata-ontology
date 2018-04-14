package main

import (
	"log"
	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/ontology"
)

func main() {

	// Get the list of result table names (as string)
	tablenames := GetTablenames()

	// Find the best partitioning based on table labels 
	coveredSets := GreedySetCover(tablenames)

	facets := GetFacetNames(coveredSets)
	log.Printf("%v", facets)
}
