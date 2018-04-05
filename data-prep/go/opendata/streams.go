package opendata

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	"hash/fnv"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// Reads the first `maxLines` lines from a file.
// Returns an array of strings, and error if any
func readLines(filename string, maxLines int) (lines []string, err error) {
	f, err := os.Open(filename)
	defer f.Close()

	if err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		value := strings.TrimSpace(scanner.Text())
		if value != "" {
			lines = append(lines, scanner.Text())
			if maxLines > 0 && maxLines <= len(lines) {
				break
			}
		}
	}
	return lines, nil
}

// Checks if a file exists or not
func exists(filename string) bool {
	_, err := os.Stat(filename)
	return err == nil
}

// Creates a channel of filenames
func StreamFilenames() <-chan string {
	output := make(chan string)
	go func() {
		f, err := os.Open(OpendataList)
		if err != nil {
			panic(err)
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			//parts := strings.Split(scanner.Text(), " ")
			//filename := filepath.Join(parts...)
			filename := scanner.Text()
			//if !strings.Contains(filename, " ") {
			output <- filename
			//}
		}
		close(output)
	}()

	return output
}

// The projected column fragment.
type Domain struct {
	Filename    string   // the logical filename of the CSV file
	Index       int      // the position of the domain in the csv file
	Values      []string // the list of values in THIS fragment.
	Cardinality int      // the cardinality of domain
	Size        int      // size of domain
}

// Saves the domain values into its value file
// and reports the status using the logger provided
func (domain *Domain) Save(logger *log.Logger) int {
	var filepath string
	dirPath := path.Join(OutputDir, "domains", domain.Filename)

	if domain.Index < 0 {
		// Encountered a file for the first time.
		// This is the header, so create the index file
		filepath = path.Join(dirPath, "index")
	} else {
		filepath = path.Join(dirPath, fmt.Sprintf("%d.values", domain.Index))
	}

	f, err := os.OpenFile(filepath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	defer f.Close()

	if err == nil {
		for _, value := range domain.Values {
			fmt.Fprintln(f, value)
		}
	} else {
		panic(fmt.Sprintf("Unable to save: %s", err.Error()))
	}

	logger.Printf("Written to %s\n", filepath)
	return len(domain.Values)
}

func (domain *Domain) PhysicalFilename(ext string) string {
	fullpath := path.Join(OutputDir, "domains", domain.Filename)

	if ext != "" {
		fullpath = path.Join(fullpath, fmt.Sprintf("%d.%s", domain.Index, ext))
	}

	return fullpath
}

func (domain *Domain) Id() string {
	return fmt.Sprintf("%s/%d", domain.Filename, domain.Index)
}

// Loads the headers from the index file
// of the csv file
func GetDomainHeader(file string) *Domain {
	filepath := path.Join(OutputDir, "domains", file, "index")
	lines, err := readLines(filepath, -1)
	if err != nil {
		return &Domain{
			Filename: file,
			Index:    -1,
		}
		//panic(err)
	}
	return &Domain{
		Filename: file,
		Index:    -1,
		Values:   lines,
	}
}

// cells - 2D array of values, a fragment of the total CSV file content
// Returns the projected domain fragments from cells
func domainsFromCells(cells [][]string, filename string, width int) []*Domain {
	if len(cells) == 0 {
		return nil
	}

	domains := make([]*Domain, width)
	for i := 0; i < width; i++ {
		domains[i] = &Domain{filename, i, nil, 0, 0}
	}
	for _, row := range cells {
		for c := 0; c < width; c++ {
			if c < len(row) {
				value := strings.TrimSpace(row[c])
				if len(value) > 2 {
					domains[c].Values = append(domains[c].Values, row[c])
				}
			}
		}
	}
	return domains
}

// A single worker function that "moves" filenames
// for the input channel to domains of the output channel
func makeDomains(filenames <-chan string, out chan *Domain) {
	for filename := range filenames {
		f, err := os.Open(Filepath(filename))
		if err != nil {
			panic(err)
			f.Close()
			continue
		}
		// Uses the csv parser to read
		// the csv content line by line
		// the first row is the headers of the domains
		rdr := csv.NewReader(f)
		header, err := rdr.Read()
		if err != nil {
			continue
		}
		width := len(header)

		headerDomain := &Domain{
			Filename: filename,
			Index:    -1,
			Values:   header,
		}

		out <- headerDomain
		var cells [][]string
		for {
			row, err := rdr.Read()
			if err == io.EOF {
				// at the end-of-file, we output the domains from the
				// cells buffer
				for _, domain := range domainsFromCells(cells, filename, width) {
					out <- domain
				}
				break
			} else {
				// read row by row into the cells buffer until there are over 1 million entries
				cells = append(cells, row)
				if len(cells)*width > 1000000 {
					for _, domain := range domainsFromCells(cells, filename, width) {
						out <- domain
					}
					cells = nil
				}
			}
		}
		f.Close()
	}
}

// Maps makeDomain to the input channel of filenames
// Uses fanout number of goroutines
func StreamDomainsFromFilenames(fanout int, filenames <-chan string) <-chan *Domain {
	out := make(chan *Domain)

	wg := &sync.WaitGroup{}

	for id := 0; id < fanout; id++ {
		wg.Add(1)
		go func(id int) {
			makeDomains(filenames, out)
			wg.Done()
		}(id)
	}
	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

type ProgressCounter struct {
	Values int
}

func Hash(s string) int {
	h := fnv.New32a()
	h.Write([]byte(s))
	return int(h.Sum32())
}

// Saves the domain fragments from an input channel to disk
// using multiple goroutines specified by fanout.
// Returns a channel of progress counter
func DoSaveDomainValues(fanout int, domains <-chan *Domain) <-chan ProgressCounter {
	progress := make(chan ProgressCounter)
	wg := &sync.WaitGroup{}

	// For each worker, have its own input queue as a channel
	// of domains
	queues := make([]chan *Domain, fanout)
	wg.Add(fanout)
	for i := 0; i < fanout; i++ {
		queues[i] = make(chan *Domain)
		// Start the goroutine and pass it
		// its own input queue
		go func(id int, queue chan *Domain) {
			logf, err := os.OpenFile(path.Join(OutputDir, "logs", fmt.Sprintf("save_domain_values_%d.log", id)),
				os.O_CREATE|os.O_WRONLY|os.O_APPEND,
				0644)
			defer logf.Close()

			if err != nil {
				panic(err)
			}
			logger := log.New(logf, fmt.Sprintf("[%d]", id), log.Lshortfile)

			for domain := range queue {
				n := domain.Save(logger)
				progress <- ProgressCounter{n}
			}
			wg.Done()
		}(i, queues[i])
	}

	// Start the router that moves domains
	// from the input channel to the individual worker input queues
	go func() {
		for domain := range domains {
			k := Hash(domain.Filename) % fanout
			queues[k] <- domain
		}
		for i := 0; i < fanout; i++ {
			close(queues[i])
		}
	}()

	go func() {
		wg.Wait()
		close(progress)
	}()

	return progress
}

// Classifies the values into one of several categories
// "numeric"
// "text"
// "" (for unknown)

var (
	patternInteger *regexp.Regexp
	patternFloat   *regexp.Regexp
	patternWord    *regexp.Regexp
)

func init() {
	patternInteger = regexp.MustCompile(`^\d+$`)
	patternFloat = regexp.MustCompile(`^\d+\.\d+$`)
	patternWord = regexp.MustCompile(`[[:alpha:]]{2,}`)
}

func isNumeric(val string) bool {
	return patternInteger.MatchString(val) || patternFloat.MatchString(val)
}

func isText(val string) bool {
	return patternWord.MatchString(val)
}

// Classifies an array of strings.  The most dominant choice
// is the class reported.
func classifyValues(values []string) string {
	var counts = make(map[string]int)

	for _, value := range values {
		var key string
		switch {
		case isNumeric(value):
			key = "numeric"
		case isText(value):
			key = "text"
		}
		if key != "" {
			counts[key] += 1
		}
	}

	var (
		maxKey   string
		maxCount int
	)
	for k, v := range counts {
		if v > maxCount {
			maxKey = k
		}
	}
	return maxKey
}

func classifyDomains(file string) {
	header := GetDomainHeader(file)
	fout, err := os.OpenFile(path.Join(OutputDir, "domains", file, "types"), os.O_CREATE|os.O_WRONLY, 0644)
	defer fout.Close()

	if err != nil {
		panic(err)
	}
	for i := 0; i < len(header.Values); i++ {
		domain_file := path.Join(OutputDir, "domains", file, fmt.Sprintf("%d.values", i))
		if !exists(domain_file) {
			continue
		}

		values, err := readLines(domain_file, 100)
		if err == nil {
			fmt.Fprintf(fout, "%d %s\n", i, classifyValues(values))
		} else {
			panic(err)
		}
	}
}

func DoClassifyDomainsFromFiles(fanout int, files <-chan string) <-chan int {
	out := make(chan int)
	wg := &sync.WaitGroup{}

	for i := 0; i < fanout; i++ {
		wg.Add(1)
		go func(id int) {
			n := 0
			for file := range files {
				classifyDomains(file)
				n += 1
				if n%100 == 0 {
					out <- n
					n = 0
				}
			}
			out <- n
			wg.Done()
		}(i)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

func getColumnNames(file string) (names []string) {
	indexFile := path.Join(OutputDir, "domains", file, "index")
	f, err := os.Open(indexFile)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		names = append(names, scanner.Text())
	}
	return
}

func getNonNumericDomains(file string) (indices []int) {
	typesFile := path.Join(OutputDir, "domains", file, "types")
	f, err := os.Open(typesFile)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.SplitN(scanner.Text(), " ", 2)
		if len(parts) == 2 {
			index, err := strconv.Atoi(parts[0])
			if err != nil {
				log.Printf("error in types of file: %s", file)
				panic(err)
			}
			if parts[1] != "numeric" {
				indices = append(indices, index)
			}
		} else {
			log.Printf("get text domains not 2: %v %s", parts, file)
		}
	}
	return
}

func getAllDomains(file string) (indices []int) {
	typesFile := path.Join(OutputDir, "domains", file, "types")
	f, err := os.Open(typesFile)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.SplitN(scanner.Text(), " ", 2)
		if len(parts) == 2 {
			index, err := strconv.Atoi(parts[0])
			if err != nil {
				log.Printf("error in types of file: %s", file)
				panic(err)
			}
			if parts[1] == "text" {
				indices = append(indices, index)
			} else {
				indices = append(indices, -1)
			}
		} else {
			log.Printf("get text domains not 2: %v %s", parts, file)
		}
	}
	return
}

func getTextDomains(file string) (indices []int) {
	typesFile := path.Join(OutputDir, "domains", file, "types")
	f, err := os.Open(typesFile)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.SplitN(scanner.Text(), " ", 2)
		if len(parts) == 2 {
			index, err := strconv.Atoi(parts[0])
			if err != nil {
				log.Printf("error in types of file: %s", typesFile)
				continue
				//panic(err)
			}
			if parts[1] == "text" {
				indices = append(indices, index)
			}
		} else {
			log.Printf("get text domains not 2: %v %s", parts, file)
		}
	}
	return
}

func normalize(w string) string {
	return strings.ToLower(w)
}

var patternSymb *regexp.Regexp

func init() {
	patternSymb = regexp.MustCompile(`[^a-z ]`)
}

func wordsFromLine(line string) []string {
	line = normalize(line)
	words := patternSymb.Split(line, -1)

	return words
}

func streamDomainWords(file string, index int, out chan *Domain) {
	filepath := path.Join(OutputDir, "domains", file, fmt.Sprintf("%d.values", index))
	f, err := os.Open(filepath)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	var values []string
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		words := wordsFromLine(scanner.Text())
		for _, word := range words {
			values = append(values, normalize(word))
			//if len(values) >= 1000 {
			//	out <- &Domain{
			//		Filename: file,
			//		Index:    index,
			//		Values:   values,
			//	}
			//	values = nil
			//}
		}
	}
	out <- &Domain{
		Filename: file,
		Index:    index,
		Values:   values,
	}
}

func StreamDomainValuesFromFiles(fanout int, files <-chan string) <-chan *Domain {
	out := make(chan *Domain)
	wg := &sync.WaitGroup{}
	wg.Add(fanout)
	for i := 0; i < fanout; i++ {
		go func(id int) {
			for file := range files {
				for _, index := range getTextDomains(file) {
					streamDomainWords(file, index, out)
				}
			}
			wg.Done()
		}(i)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

type ValueFreq struct {
	Filename string
	Index    int
	Values   []string
	Freq     []int
}

func (vf *ValueFreq) String() string {
	var buf bytes.Buffer
	for i := 0; i < len(vf.Values); i++ {
		v := vf.Values[i]
		f := vf.Freq[i]
		fmt.Fprintf(&buf, "%s: %d\n", v, f)
	}
	return buf.String()
}

func getValueFreq(filename, datafilename string, index int) *ValueFreq {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	values := strings.Split(string(content), "\n")
	freq := make(map[string]int)
	for _, value := range values {
		freq[value] += 1
	}
	n := len(freq)

	valuefreq := &ValueFreq{
		Filename: datafilename,
		Index:    index,
		Values:   make([]string, n),
		Freq:     make([]int, n),
	}

	i := 0
	for k, v := range freq {
		valuefreq.Values[i] = k
		valuefreq.Freq[i] = v
		i += 1
	}

	return valuefreq
}

func StreamValueFreqFromCache(fanout int, filenames <-chan string) <-chan *ValueFreq {
	out := make(chan *ValueFreq)
	wg := &sync.WaitGroup{}
	for i := 0; i < fanout; i++ {
		wg.Add(1)
		go func(id int, out chan<- *ValueFreq) {
			for filename := range filenames {
				for _, index := range getTextDomains(filename) {
					//fp := path.Join(OutputDir, "domains", filename, fmt.Sprintf("%d.ft-sum", index))
					//f, err := os.Open(fp)
					//f.Close()
					//if err != nil {
					//	log.Printf("should generate emb for %s", fp)
					d := &Domain{
						Filename: filename,
						Index:    index,
					}
					valueFilename := d.PhysicalFilename("values")
					out <- getValueFreq(valueFilename, filename, index)
					//}
				}
			}
			wg.Done()
		}(i, out)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

func StreamAllODEmbVectors(fanout int, filenames <-chan string) <-chan string {
	dfilename := "us.data.gov/t_0033d0e05a7f31a3.csv"
	dindex := 0
	out := make(chan string)
	wg := &sync.WaitGroup{}
	wg.Add(fanout)
	for i := 0; i < fanout; i++ {
		go func(id int, out chan<- string) {
			for filename := range filenames {
				if strings.Contains(filename, "us.data.gov") == true {
					for _, index := range getAllDomains(filename) {
						d := &Domain{Filename: "", Index: 0}
						if index == -1 {
							d = &Domain{
								Filename: dfilename,
								Index:    dindex,
							}
						} else {
							d = &Domain{
								Filename: filename,
								Index:    index,
							}
						}
						embFilename := d.PhysicalFilename("ft-mean")
						out <- embFilename
					}
				} else {
					for _, index := range getTextDomains(filename) {
						d := &Domain{
							Filename: filename,
							Index:    index,
						}
						embFilename := d.PhysicalFilename("ft-mean")
						out <- embFilename
					}
				}
			}
			wg.Done()
		}(i, out)

	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

func StreamEmbVectors(fanout int, filenames <-chan string) <-chan string {
	out := make(chan string)
	wg := &sync.WaitGroup{}
	wg.Add(fanout)
	for i := 0; i < fanout; i++ {
		go func(id int, out chan<- string) {
			for filename := range filenames {
				//for _, index := range getNonNumericDomains(filename) {
				for _, index := range getTextDomains(filename) {
					d := &Domain{
						Filename: filename,
						Index:    index,
					}
					//embFilename := d.PhysicalFilename("ft-sum")
					embFilename := d.PhysicalFilename("ft-mean")
					out <- embFilename
				}
			}
			wg.Done()
		}(i, out)

	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

// Creates a channel of filenames
func StreamQueryFilenames() <-chan string {
	output := make(chan string)
	go func() {
		f, err := os.Open(QueryList)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			parts := strings.SplitN(scanner.Text(), " ", 3)
			filename := path.Join(parts...)
			//filename := scanner.Text()
			output <- filename
		}
		close(output)
	}()

	return output
}
