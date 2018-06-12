package opendata

import (
	"os"
	"strconv"
)

func WriteMinhashToDisk(vec []uint64, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	line := ""
	for i, e := range vec {
		f := strconv.FormatFloat(float64(e), 'f', 2, 64)
		if i != (len(vec)-1) {
			line += f + "," 
		} else {
			line += f
		}
	}
	if _, err := file.WriteString(line); err != nil {
		return err
	}
	return nil
}

func WriteEmbVecToDisk(vec []float64, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	line := ""
	for i, e := range vec {
		f := strconv.FormatFloat(e, 'f', 6, 64)
		if i != (len(vec)-1) {
			line += f + "," 
		} else {
			line += f
		}
	}
	if _, err := file.WriteString(line); err != nil {
		return err
	}
	return nil
}
