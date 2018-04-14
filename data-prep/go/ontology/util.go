package ontology

import (
	"os"
	"bufio"
	"io/ioutil"
	"encoding/json"
	"strconv"
)

// Creates a channel of table names
func GetTablenames() []string {
	output := make([]string, 0)
	f, err := os.Open(QueryResultList)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		tablename := scanner.Text()
		output = append(output, tablename)
	}
	return output
}

func GetFacetNames(labels map[string]map[string]float64) []string {
	labelNames := make(map[string]int)
	b, err := ioutil.ReadFile(LabelsFile)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(b, &labelNames)
	if err != nil {
		panic(err)
	}
	reverseLabels := make(map[int]string)
	for k, v := range labelNames {
		reverseLabels[v] = k
	}
	names := make([]string, 0)
	for l, _ := range labels{
		l, _ := strconv.Atoi(l)
		names = append(names, reverseLabels[l])
	}
	return names
}
