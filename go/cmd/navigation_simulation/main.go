package main

import (
	"log"

	. "github.com/RJMillerLab/opendata-ontology/go/organization"
)

func main() {
	//org := NewOrganization()
	//org.InitializeNavigationPlus()
	//org.GenerateRuns(30000)
	//org.ProcessRuns()
	Initialize()
	orgs := GenerateOrganizations(20)
	//ODTransitions()
	//org := ReadOrganization()
	for i, org := range orgs {
		log.Printf("evaluating org %d.", i)
		orgSuccessProb := EvaluateOrganization(org, 50)
		log.Printf("the prob of success of org: %f", orgSuccessProb)
	}
}
