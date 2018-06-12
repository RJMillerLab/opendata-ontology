package old_rganization

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

type OrgEval struct {
	Org               [][]int
	Nonuniformity     int
	Entropy           float64
	Density           int
	IntraDisjointness int
	Score             float64
}

func ReadOrganzations() <-chan [][]int {
	output := make(chan [][]int)
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		f, err := os.Open(OrgsFile)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			parts := strings.SplitN(line, ",", -1)
			org := make([][]int, 2)
			for i, p := range parts {
				ip, _ := strconv.Atoi(p)
				if ip == 0 {
					org[0] = append(org[0], i)
				} else {
					org[1] = append(org[1], i)
				}
			}
			output <- org
		}
		wg.Done()
	}()
	go func() {
		wg.Wait()
		close(output)
	}()
	return output
}

func ComputeLabelOverlaps() map[int]map[int]int {
	goodLabels := make([]int, 0)
	labelNames := make(map[string]string)
	labelTables := make(map[string][]string)
	err := loadJson(GoodLabelsFile, &goodLabels)
	if err != nil {
		panic(err)
	}
	goodLabels = goodLabels[:20]
	err = loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	gln := make([]string, 0)
	for _, gl := range goodLabels {
		gln = append(gln, labelNames[strconv.Itoa(gl)])
	}
	log.Println("good lables: %v", gln)
	err = loadJson(LabelTablesFile, &labelTables)
	if err != nil {
		panic(err)
	}
	overlaps := make(map[int]map[int]int)
	for il1 := 0; il1 < (len(goodLabels) - 1); il1++ {
		overlaps[goodLabels[il1]] = make(map[int]int)
		for il2 := il1; il2 < len(goodLabels); il2 += 1 {
			if il1 == il2 {
				continue
			}
			co := computeOverlapSize(labelTables[strconv.Itoa(goodLabels[il1])], labelTables[strconv.Itoa(goodLabels[il2])])
			if co != 0 {
				overlaps[goodLabels[il1]][goodLabels[il2]] = co
			}
		}
	}
	return overlaps
}

func computeOverlapSize(t1s, t2s []string) int {
	overlap := 0
	for _, t1 := range t1s {
		if ContainsStr(t2s, t1) == true {
			overlap += 1
		}
	}
	return overlap
}

func FindOrganization(orgs <-chan [][]int, overlaps map[int]map[int]int) ([]OrgEval, []OrgEval, []OrgEval) {
	results1 := make(chan OrgEval)
	results2 := make(chan OrgEval)
	wg := &sync.WaitGroup{}
	wg.Add(35)
	for i := 0; i < 35; i++ {
		go func() {
			for org := range orgs {
				density := evaluateOrganizationDensity(org, overlaps)
				if density == 0 {
					continue
				}
				nonuniformity := evaluateOrganizationUniformityDiff(org, overlaps)
				entropy := evaluateOrganizationUniformityEntropy(org, overlaps, density)
				oe := OrgEval{
					Org:               org,
					Density:           density,
					Nonuniformity:     nonuniformity,
					Entropy:           entropy,
					IntraDisjointness: evaluateIntradimension(org, overlaps),
					Score:             float64(density) / float64(1+nonuniformity),
				}
				results1 <- oe
				results2 <- oe
			}
			wg.Done()
		}()
	}
	go func() {
		serializeOrganizations(results1)
	}()
	go func() {
		wg.Wait()
		close(results1)
		close(results2)
	}()
	bestOrgsDensity, bestOrgsUniformity, bestOrgsAgg := findBestScores(results2)
	return bestOrgsDensity, bestOrgsUniformity, bestOrgsAgg
}

func serializeOrganizations(results chan OrgEval) {
	// ignoring all errors of IO
	labelNames := make(map[string]string)
	err := loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	f, _ := os.Create(OrgsFile)
	defer f.Close()
	// each 2D org is serialized as three lines
	// one line for labels in each dim (label_id|label_name)
	// the last line is for scores: density, nonuniformity, entropy, disjointness
	for orgEval := range results {
		for _, ls := range orgEval.Org {
			for i, l := range ls {
				if i != 0 {
					f.WriteString("|" + strconv.Itoa(l) + "|" + labelNames[strconv.Itoa(l)])
				} else {
					f.WriteString(strconv.Itoa(l) + "|" + labelNames[strconv.Itoa(l)])
				}
			}
			f.WriteString("\n")
		}
		f.WriteString(strconv.Itoa(orgEval.Density) + "|" + strconv.Itoa(orgEval.Nonuniformity) + "|" + strconv.FormatFloat(orgEval.Entropy, 'f', -1, 64) + "|" + strconv.Itoa(orgEval.IntraDisjointness))
		f.WriteString("\n")
	}
}

func findBestScores(results chan OrgEval) ([]OrgEval, []OrgEval, []OrgEval) {
	resultCount := -1
	bestOrgAggScore := -1.0
	bestOrgDensityScore := -1
	bestOrgUniformityScore := -1
	bestOrgsDensity := make([]OrgEval, 0)
	bestOrgsUniformity := make([]OrgEval, 0)
	bestOrgsAgg := make([]OrgEval, 0)
	for oe := range results {
		resultCount += 1
		if resultCount%1000 == 0 {
			log.Printf("processed %d orgs and the best score is %f", resultCount, bestOrgAggScore)
		}
		if oe.Score == bestOrgAggScore {
			bestOrgsAgg = append(bestOrgsAgg, oe)
		} else if oe.Score > bestOrgAggScore {
			bestOrgsAgg = make([]OrgEval, 0)
			bestOrgsAgg = append(bestOrgsAgg, oe)
			bestOrgAggScore = oe.Score
		}
		if oe.Density == bestOrgDensityScore {
			bestOrgsDensity = append(bestOrgsDensity, oe)
		} else if oe.Density > bestOrgDensityScore {
			bestOrgsDensity = make([]OrgEval, 0)
			bestOrgsDensity = append(bestOrgsDensity, oe)
			bestOrgDensityScore = oe.Density
		}
		if oe.Nonuniformity == bestOrgUniformityScore {
			bestOrgsUniformity = append(bestOrgsUniformity, oe)
		} else if oe.Nonuniformity < bestOrgUniformityScore {
			bestOrgsUniformity = make([]OrgEval, 0)
			bestOrgsUniformity = append(bestOrgsUniformity, oe)
			bestOrgUniformityScore = oe.Nonuniformity
		}
	}
	return bestOrgsDensity, bestOrgsUniformity, bestOrgsAgg
}

func evaluateOrganizationUniformityDiff(org [][]int, overlaps map[int]map[int]int) int {
	dim1 := org[0]
	dim2 := org[1]
	nonuniformity := 0.0
	for _, l1 := range dim1 {
		diff := 0.0
		for il2, l2 := range dim2 {
			for il3 := il2 + 1; il3 < len(dim2); il3++ {
				l3 := dim2[il3]
				diff += math.Abs(float64(getOverlap(overlaps, l1, l2) - getOverlap(overlaps, l1, l3)))
			}
		}
		nonuniformity += diff
	}
	return int(nonuniformity)
}

func evaluateOrganizationUniformityEntropy(org [][]int, overlaps map[int]map[int]int, density int) float64 {
	dim1 := org[0]
	dim2 := org[1]
	entropy := 0.0
	for _, l1 := range dim1 {
		for _, l2 := range dim2 {
			p := float64(getOverlap(overlaps, l1, l2)) / float64(density)
			if p != 0 {
				entropy -= p * math.Log(p)
			}
		}
	}
	return entropy
}

func evaluateIntradimension(org [][]int, overlaps map[int]map[int]int) int {
	disjointness := 0
	for _, ls := range org {
		for i, l1 := range ls {
			for j := i + 1; j < len(ls); j++ {
				l2 := ls[j]
				disjointness += getOverlap(overlaps, l1, l2)
			}
		}
	}
	return disjointness
}

func getOverlap(overlaps map[int]map[int]int, l1, l2 int) int {
	if _, ok := overlaps[l1][l2]; !ok {
		if _, ok := overlaps[l2][l1]; !ok {
			return 0
		} else {
			return overlaps[l2][l1]
		}
	}
	return overlaps[l1][l2]
}

func evaluateOrganizationDensity(org [][]int, overlaps map[int]map[int]int) int {
	dim1 := org[0]
	dim2 := org[1]
	score := 0
	for _, l1 := range dim1 {
		for _, l2 := range dim2 {
			score += getOverlap(overlaps, l1, l2)
		}
	}
	return score
}

func Generate2DimOrganizations() <-chan [][]int {
	// this function works with the index of labels
	orgs := make(chan [][]int, 500)
	go func() {
		goodLabels := make([]int, 0)
		err := loadJson(GoodLabelsFile, &goodLabels)
		if err != nil {
			panic(err)
		}
		goodLabels = goodLabels[:20]
		labelsNum := len(goodLabels)
		orgCount := int(math.Pow(2, float64(labelsNum)))
		//orgs := make([][][]int, 0)
		genOrgCount := -1
		for i := 1; i < orgCount/2; i++ {
			org := [][]int{[]int{}, []int{}}
			bs := strconv.FormatUint(uint64(i), 2)
			ps := strings.SplitN(bs, "", -1)
			for k := labelsNum - 1; k > (len(ps) - 1); k-- {
				org[0] = append(org[0], goodLabels[k])
			}
			for j, p := range ps {
				pi, _ := strconv.Atoi(p)
				if pi == 0 {
					org[0] = append(org[0], goodLabels[len(ps)-j-1])
				} else {
					org[1] = append(org[1], goodLabels[len(ps)-j-1])
				}
			}
			if len(org[0]) > 1 && len(org[1]) > 1 {
				genOrgCount += 1
				orgs <- org
			}
			if genOrgCount > 0 && genOrgCount%1000 == 0 {
				log.Printf("generated %d orgs", genOrgCount)
			}
		}
		close(orgs)
	}()
	return orgs
}

func PrintOrganization(orgEvals []OrgEval, overlaps map[int]map[int]int) {
	labelNames := make(map[string]string)
	err := loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	for i, orgEval := range orgEvals {
		fmt.Printf("Organization %d with density %d and non-uniformity %d and total score %f\n", i, orgEval.Density, orgEval.Nonuniformity, orgEval.Score)
		fmt.Println("-----------------------------------------------------------------")
		org := orgEval.Org
		d1names := make([]string, 0)
		//d1names := make([]int, 0)
		for _, l := range org[0] {
			d1names = append(d1names, labelNames[strconv.Itoa(l)])
			//d1names = append(d1names, l)
		}
		fmt.Println(d1names)
		for _, l := range org[1] {
			//fmt.Printf("%d", l)
			fmt.Printf("%s", labelNames[strconv.Itoa(l)])
			os := make([]int, 0)
			for _, k := range org[0] {
				os = append(os, getOverlap(overlaps, l, k))
			}
			fmt.Println(os)
		}
		fmt.Println("disjoitness: %d", orgEval.IntraDisjointness)
		fmt.Println("-----------------------------------------------------------------")
	}
}
