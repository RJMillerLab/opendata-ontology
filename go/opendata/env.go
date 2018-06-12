package opendata

import (
	"os"
	"path"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const PARALLEL = 64
const MIN_DOMSIZE = 5

// Environment variables required to
// locate the necessary input files
var OpendataDir = os.Getenv("OPENDATA_DIR")
var OpendataList = os.Getenv("OPENDATA_LIST")

// Environment variable required to
// write output
var OutputDir = os.Getenv("OUTPUT_DIR")

// Environment variable for domain pairs unionability scores
var QueryList = os.Getenv("QUERY_LIST")

func CheckEnv() {
}

func Filepath(filename string) string {
	return path.Join(OpendataDir, "files", filename)
}

func GetNow() float64 {
	return float64(time.Now().UnixNano()) / 1E9
}

func GenericStrings(s []string) []interface{} {
	var a = make([]interface{}, len(s))
	for i := 0; i < len(s); i++ {
		a[i] = s[i]
	}
	return a
}
