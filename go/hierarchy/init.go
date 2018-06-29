package hierarchy

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
	"unicode"

	. "github.com/RJMillerLab/opendata-ontology/go/embedding"
)

var (
	tableFileExt = ".csv"
	tagFileExt   = ".tags.vec"
	ftConn       *FastText
)

type domain struct {
	tablename string   // the logical filename of the CSV file
	index     int      // the position of the domain in the csv file
	values    []string // the list of values
	sem       []float64
	tags      []string
}

func buildContext() (map[string][]float64, map[string]string, map[string][]string, map[string][]string, map[string][]string, map[string][][]float64, map[string][]float64) {
	ftConn = initFasttextConn()
	tables := readTablenames()
	domainSems := make(map[string][]float64)
	domainTags := make(map[string]string)
	tableTags := make(map[string][]string)
	tagDomains := make(map[string][]string)
	tagTables := make(map[string][]string)
	tagDomainSems := make(map[string][][]float64)
	for _, tablename := range tables {
		domains := readDomains(tablename)
		tags := readTags(tablename)
		tableTags[tablename] = make([]string, 0)
		for i, domain := range domains {
			domainName := domain.getName()
			domain.getSem()
			domainSems[domainName] = domain.sem
			// one tag per domain
			domainTags[domainName] = tags[i]
			tableTags[tablename] = append(tableTags[tablename], tags[i])
			if _, ok := tagDomains[tags[i]]; !ok {
				tagDomainSems[tags[i]] = make([][]float64, 0)
				tagDomains[tags[i]] = make([]string, 0)
			}
			if _, ok := tagTables[tags[i]]; !ok {
				tagTables[tags[i]] = make([]string, 0)
			}
			tagDomainSems[tags[i]] = append(tagDomainSems[tags[i]], domain.sem)
			tagDomains[tags[i]] = append(tagDomains[tags[i]], domainName)
			tagTables[tags[i]] = append(tagTables[tags[i]], tablename)
		}
	}
	log.Printf("domainSems: %d domainTags: %d tableTags: %d tagDomains: %d tagTables: %d tagDomainSems: %d tagSems: %d", len(domainSems), len(domainTags), len(tableTags), len(tagDomains), len(tagTables), len(tagDomainSems), len(tagSems))
	tagSems := getTagSems(tagDomains, domainSems)
	return domainSems, domainTags, tableTags, tagDomains, tagTables, tagDomainSems, tagSems
}

func readTablenames() []string {
	tables := make([]string, 0)
	f, err := os.Open(TablesFile)
	defer f.Close()
	if err != nil {
		panic(err)
	}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		tables = append(tables, line)
	}
	return tables
}

func (domain *domain) getSem() {
	sem, err := ftConn.GetDomainEmbMeanNoFreq(domain.values)
	log.Printf("sem: %v", sem)
	if err != nil {
		panic(err)
	}
	domain.sem = sem
}

func readDomains(tablename string) []*domain {
	domains := make([]*domain, 0)
	f, err := os.Open(path.Join(TablesDir, tablename+tableFileExt))
	if err != nil {
		panic(err)
		f.Close()
	}
	var cells [][]string
	rdr := csv.NewReader(f)
	for {
		row, err := rdr.Read()
		if err == io.EOF {
			domains = domainsFromCells(cells, tablename)
			break
		} else {
			cells = append(cells, row)
		}
	}
	return domains
}

func readTags(tablename string) []string {
	tags := make([]string, 0)
	f, err := os.Open(path.Join(TablesDir, tablename+tagFileExt))
	defer f.Close()
	if err != nil {
		panic(err)
	}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		tag := strings.Split(line, " ")[0]
		tags = append(tags, tag)
	}
	return tags
}

func getTagSems(tagDomains map[string][]string, domainSems map[string][]float64) map[string][]float64 {
	tagSems := make(map[string][]float64)
	for t, ds := range tagDomains {
		vecs := make([][]float64, 0)
		for _, d := range ds {
			vecs = append(vecs, domainSems[d])
		}
		tagSems[t] = avg(vecs)
	}
	return tagSems
}

// cells - 2D array of values
// Returns the projected domain fragments from cells
func domainsFromCells(cells [][]string, filename string) []*domain {
	if len(cells) == 0 {
		return nil
	}
	width := len(cells[0])
	domains := make([]*domain, width)
	for i := 0; i < width; i++ {
		domains[i] = &domain{filename, i, make([]string, 0), make([]float64, 0), make([]string, 0)}
	}
	for _, row := range cells {
		for c := 0; c < width; c++ {
			if c < len(row) {
				value := strings.TrimSpace(row[c])
				if len(value) > 2 {
					domains[c].values = append(domains[c].values, row[c])
				}
			}
		}
	}
	return domains
}

func (domain *domain) getName() string {
	return domain.tablename + "_" + strconv.Itoa(domain.index)
}

func initFasttextConn() *FastText {
	ft, err := InitInMemoryFastText(FasttextDb, func(v string) []string {
		stopWords := []string{"ckan_topiccategory_", "ckan_keywords_", "ckan_tags_", "ckan_subject_", "socrata_domaincategory_", "socrata_domaintags_", "socrata_tags_"}
		for _, st := range stopWords {
			v = strings.Replace(v, st, "", -1)
		}
		v = strings.Replace(strings.Replace(strings.Replace(v, "_", " ", -1), "-", " ", -1), "\\'", " ", -1)
		return strings.Split(v, " ")
	}, func(v string) string {
		return strings.ToLower(strings.TrimFunc(strings.TrimSpace(v), unicode.IsPunct))
	})
	if err != nil {
		panic(err)
	}
	return ft
}
