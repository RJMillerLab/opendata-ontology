package ontology

import (
	"bufio"
	"encoding/json"
	"io/ioutil"
	"os"
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
	for l, _ := range labels {
		l, _ := strconv.Atoi(l)
		names = append(names, reverseLabels[l])
	}
	return names
}

func loadJson(file string, v interface{}) (err error) {
	buffer, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	err = json.Unmarshal(buffer, v)
	if err != nil {
		return err
	}
	return nil
}

func ContainsInt(as []int, i int) bool {
	for _, v := range as {
		if v == i {
			return true
		}
	}
	return false
}

func ContainsStr(as []string, i string) bool {
	for _, v := range as {
		if v == i {
			return true
		}
	}
	return false
}
