package ontology

import (
	"os"
	"bufio"
	"io/ioutil"
	"encoding/json"
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

func GetLabelNames(labels []string) []string {
	labelNames := make(map[int]string)
	b, err := ioutil.ReadFile(LabelsFile)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(b, &labelNames)
	if err != nil {
		panic(err)
	}
	names := make([]string, 0)
	for l := range labels{
		names = append(names, labelNames[l])
	}
	return names
}
