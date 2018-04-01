package opendata

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"strconv"
)

func WriteVec(vec []float64, order binary.ByteOrder, file io.Writer) error {
	for _, v := range vec {
		err := binary.Write(file, order, v)
		if err != nil {
			return err
		}
	}
	return nil
}

func WriteVecToDisk(vec []float64, filename string) error {
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

func ReadVecFromDisk(filename string, order binary.ByteOrder) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	stats, serr := file.Stat()
	if serr != nil {
		return nil, serr
	}
	var size int64 = stats.Size()
	binVec := make([]byte, size)
	if _, rerr := file.Read(binVec); rerr != nil {
		return nil, rerr
	}
	vec, verr := BytesToVec(binVec, order)
	return vec, verr
}

func Cosine(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("Length of vectors not equal")
	}
	dot := 0.0
	modX, modY := 0.0, 0.0
	for i := range x {
		dot += x[i] * y[i]
		modX += x[i] * x[i]
		modY += y[i] * y[i]
	}
	return dot / (math.Sqrt(modX) * math.Sqrt(modY))
}
