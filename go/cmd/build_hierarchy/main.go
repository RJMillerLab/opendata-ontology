package main

import (
	. "github.com/RJMillerLab/opendata-ontology/go/hierarchy"
)

func main() {
	clustering := Initialize()
	//clustering.MakeThresholdClustering()
	clustering.BuildClusters()
	//org := clustering.ToOrganization()
	//org.Evaluate()
}
