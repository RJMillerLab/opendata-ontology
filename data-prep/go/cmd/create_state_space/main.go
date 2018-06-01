package main

import (
	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/space"
)

func main() {
	clustering := Initialize()
	clustering.BuildClusters()
}
