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
	//ODTransitions()
	//org := ReadOrganization()
	Initialize()
	orgs := GenerateOrganizations(5)
	bestOrg := orgs[0]
	bestProb := EvaluateOrganization(bestOrg, 50)
	for i, org := range orgs {
		log.Printf("evaluating org %d.", i)
		orgSuccessProb := EvaluateOrganization(org, 50)
		log.Printf("the prob of success of org: %f", orgSuccessProb)
		if bestProb < orgSuccessProb {
			bestProb = orgSuccessProb
			bestOrg = org
		}
	}
	bestOrg.Print()
	log.Printf("success prob: %f", bestProb)
}
