package main

import (
	"log"

	. "github.com/RJMillerLab/opendata-ontology/go/organization"
)

func main() {
	Init()
	tagSem, tagDatasets, datasetEmbs, datasetTags := SynthesizeMetadata(5, 150, 25)
	//tagSem, tagDatasets, datasetEmbs, datasetTags := SynthesizeMetadata(5, 50) //15,50
	orgs := SynthesizeOrganizations(-1, tagSem, tagDatasets, datasetEmbs, datasetTags)
	log.Printf("number of generated orgs: %v", len(orgs))
	InitSpace(tagDatasets, tagSem, datasetEmbs)
	EvaluateOrganizations(orgs, 50)
}
