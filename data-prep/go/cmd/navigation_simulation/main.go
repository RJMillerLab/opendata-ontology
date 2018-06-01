package main

import (
	. "github.com/RJMillerLab/opendata-ontology/data-prep/go/organization"
)

func main() {
	org := NewOrganization()
	org.InitializeNavigationPlus()
	//SimulatePlus()
	org.GenerateRuns(30000)
	org.ProcessRuns()
}
