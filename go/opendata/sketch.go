package opendata

import (
	"bufio"
	"fmt"
	"os"
	"path"
	"sync"

	minhashlsh "github.com/RJMillerLab/table-union/minhashlsh"
)

var (
	seed    = 1
	numHash = 256
)

type DomainSketch struct {
	Filename string              // the logical filename of the CSV file
	Index    int                 // the position of the domain in the csv file
	Sketch   *minhashlsh.Minhash // the minhash sketch
}

func DoMinhashNumericDomainsFromFiles(fanout int, files <-chan string, ext string) <-chan *DomainSketch {
	out := make(chan *DomainSketch)
	wg := &sync.WaitGroup{}

	for i := 0; i < fanout; i++ {
		wg.Add(1)
		go func(id int) {
			for file := range files {
				for _, index := range getNumericDomains(file) {
					minhashDomainWords(file, index, out, ext)
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

func minhashDomainWords(file string, index int, out chan *DomainSketch, ext string) {
	filepath := path.Join(OutputDir, "domains", file, fmt.Sprintf("%d.%s", index, ext))
	f, err := os.Open(filepath)
	if err != nil {
		return
		//panic(err)
	}
	defer f.Close()
	mh := minhashlsh.NewMinhash(seed, numHash)
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		mh.Push([]byte(normalize(scanner.Text())))
	}
	out <- &DomainSketch{
		Filename: file,
		Index:    index,
		Sketch:   mh,
	}
}

// Saves the domain skecthes from an input channel to disk
// Returns a channel of progress counter
func DoSaveDomainSketches(fanout int, sketches <-chan *DomainSketch, ext string) <-chan ProgressCounter {
	progress := make(chan ProgressCounter)
	wg := &sync.WaitGroup{}
	wg.Add(fanout)
	for i := 0; i < fanout; i++ {
		go func(id int, sketches <-chan *DomainSketch) {
			for domain := range sketches {
				minhashFilename := domain.PhysicalFilename(ext)
				err := WriteMinhashToDisk(domain.Sketch.Signature(), minhashFilename)
				if err == nil {
					progress <- ProgressCounter{1}
				}
			}
			wg.Done()
		}(i, sketches)
	}
	go func() {
		wg.Wait()
		close(progress)
	}()
	return progress
}

func (domain *DomainSketch) PhysicalFilename(ext string) string {
	fullpath := path.Join(OutputDir, "domains", domain.Filename)

	if ext != "" {
		fullpath = path.Join(fullpath, fmt.Sprintf("%d.%s", domain.Index, ext))
	}

	return fullpath
}
