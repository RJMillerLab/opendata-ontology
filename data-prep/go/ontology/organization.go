package ontology

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

type OrgEval struct {
	Org   [][]int
	Score int
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
	err = loadJson(LabelNamesFile, &labelNames)
	if err != nil {
		panic(err)
	}
	err = loadJson(LabelTablesFile, &labelTables)
	if err != nil {
		panic(err)
	}
	overlaps := make(map[int]map[int]int)
	for il1 := 0; il1 < (len(goodLabels) - 1); il1++ {
		if il1 > 10 {
			continue
		}
		overlaps[goodLabels[il1]] = make(map[int]int)
		for il2 := il1; il2 < len(goodLabels); il2 += 1 {
			if il2 > 10 {
				continue
			}
			if il1 == il2 {
				continue
			}
			overlaps[goodLabels[il1]][goodLabels[il2]] = computeOverlapSize(labelTables[strconv.Itoa(goodLabels[il1])], labelTables[strconv.Itoa(goodLabels[il2])])
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

func FindBestOrganization(orgs <-chan [][]int, overlaps map[int]map[int]int) [][]int {
	results := make(chan OrgEval)
	wg := &sync.WaitGroup{}
	wg.Add(20)
	for i := 0; i < 20; i++ {
		go func() {
			for org := range orgs {
				oe := OrgEval{
					Org:   org,
					Score: evaluateOrganization(org, overlaps),
				}
				results <- oe
			}
			wg.Done()
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()
	bestOrgScore := -1
	bestOrg := make([][]int, 0)
	for oe := range results {
		if oe.Score > bestOrgScore {
			bestOrgScore = oe.Score
			bestOrg = oe.Org
		}
	}
	return bestOrg
}

func evaluateOrganization(org [][]int, overlaps map[int]map[int]int) int {
	dim1 := org[0]
	dim2 := org[1]
	score := 0
	for _, l1 := range dim1 {
		for _, l2 := range dim2 {
			score += overlaps[l1][l2]
		}
	}
	return score
}

func Generate2DimOrganizations(labelsNum int) [][][]int {
	// this function works with the index of labels
	orgCount := int(math.Pow(2, float64(labelsNum)))
	orgs := make([][][]int, 0)
	for i := 1; i < orgCount/2; i++ {
		org := [][]int{[]int{}, []int{}}
		bs := strconv.FormatInt(int64(i), 2)
		ps := strings.SplitN(bs, "", -1)
		for k := labelsNum - 1; k > (len(ps) - 1); k-- {
			org[0] = append(org[0], k)
		}
		for j, p := range ps {
			pi, _ := strconv.Atoi(p)
			if pi == 0 {
				org[0] = append(org[0], len(ps)-j-1)
			} else {
				org[1] = append(org[1], len(ps)-j-1)
			}
		}
		orgs = append(orgs, org)
	}
	log.Println(orgs)
	return orgs
}
