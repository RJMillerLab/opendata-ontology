package opendata

import (
	"bytes"
	"encoding/binary"
	"errors"
	"log"

	"github.com/ekzhu/counter"
	fasttext "github.com/ekzhu/go-fasttext"
	"gonum.org/v1/gonum/mat"
)

var (
	ErrNoEmbFound = errors.New("No embedding found")
)

// Get the embedding vector of a column by taking the average of the distinct values (tokenized) vectors.
func GetDomainEmbAve(ft *fasttext.FastText, tokenFun func(string) []string, transFun func(string) string, column []string) ([]float64, error) {
	values := TokenizedValues(column, tokenFun, transFun)
	var vec []float64
	var count int
	for tokens := range values {
		valueVec, err := GetValueEmb(ft, tokens)
		if err != nil {
			continue
		}
		if vec == nil {
			vec = valueVec
		} else {
			add(vec, valueVec)
		}
		count++
	}
	if vec == nil {
		return nil, ErrNoEmbFound
	}
	for i, v := range vec {
		vec[i] = v / float64(count)
	}
	return vec, nil
}

// Get the embedding vector of a column by taking the sum of the values (tokenized) vectors.
func GetDomainEmbSum(ft *fasttext.FastText, tokenFun func(string) []string, transFun func(string) string, column []string) ([]float64, error) {
	values := TokenizedValues(column, tokenFun, transFun)
	var vec []float64
	for tokens := range values {
		valueVec, err := GetValueEmb(ft, tokens)
		if err != nil {
			continue
		}
		if vec == nil {
			vec = valueVec
		} else {
			add(vec, valueVec)
		}
	}
	if vec == nil {
		return nil, ErrNoEmbFound
	}
	return vec, nil
}

// Returns the mean of domain embedding matrix
func GetDomainEmbMeanCovar(ft *fasttext.FastText, tokenFun func(string) []string, transFun func(string) string, column []string) ([]float64, []float64, error) {
	values := TokenizedValues(column, tokenFun, transFun)
	dim := 300
	log.Printf("domain size: %d", len(values))
	sum := make([]float64, dim)
	mean := make([]float64, dim)
	covarSum := make([][]float64, dim)
	covar := make([][]float64, dim)
	// initialize covar matrix
	for i, _ := range covarSum {
		vec := make([]float64, dim)
		covarSum[i] = vec
		covar[i] = vec
	}
	ftValuesNum := 0
	for tokens := range values {
		vec, err := GetValueEmb(ft, tokens)
		if err != nil {
			continue
		}
		for j := 0; j < dim; j++ {
			sum[j] += vec[j]
			for k := 0; k < dim; k++ {
				covarSum[j][k] += vec[j] * vec[k]
			}
		}
	}

	for j := 0; j < dim; j++ {
		mean[j] = sum[j] / float64(ftValuesNum)
		for k := 0; k < dim; k++ {
			covarSum[j][k] = covarSum[j][k] / float64(ftValuesNum)
		}
	}

	for j := 0; j < dim; j++ {
		for k := 0; k < dim; k++ {
			covar[j][k] = covarSum[j][k] - sum[j]*sum[k]
		}
	}

	return mean, flatten2DSlice(covar), nil
}

func GetDomainEmb(ft *fasttext.FastText, tokenFun func(string) []string, transFun func(string) string, column []string, kfirst int) (*mat.Dense, error) {
	values := TokenizedValues(column, tokenFun, transFun)
	var data []float64
	var count int
	for tokens := range values {
		valueVec, err := GetValueEmb(ft, tokens)
		if err != nil {
			continue
		}
		if data == nil {
			data = valueVec
		} else {
			data = append(data, valueVec...)
		}
		count++
	}
	if count == 0 {
		return nil, ErrNoEmbFound
	}
	matrix := mat.NewDense(count, fasttext.Dim, data)
	return matrix, nil
}


// Get the embedding vector of a tokenized value by sum all the tokens' vectors
func GetValueEmb(ft *fasttext.FastText, tokenizedValue []string) ([]float64, error) {
	var valueVec []float64
	var count int
	for _, token := range tokenizedValue {
		emb, err := ft.GetEmb(token)
		if err == fasttext.ErrNoEmbFound {
			continue
		}
		if err != nil {
			panic(err)
		}
		if valueVec == nil {
			valueVec = emb
		} else {
			add(valueVec, emb)
		}
		count++
	}
	if valueVec == nil {
		return nil, ErrNoEmbFound
	}
	return valueVec, nil
}

// Produce a channel of values (tokenized)
func TokenizedValues(values []string, tokenFun func(string) []string, transFun func(string) string) chan []string {
	out := make(chan []string)
	go func() {
		for _, v := range values {
			v = transFun(v)
			// Tokenize
			tokens := tokenFun(v)
			if len(tokens) > 5 {
				// Skip text values
				continue
			}
			for i, t := range tokens {
				tokens[i] = transFun(t)
			}
			out <- tokens
		}
		close(out)
	}()
	return out
}

// Produce a channel of distinct values (tokenized)
func TokenizedDistinctValues(values []string, tokenFun func(string) []string, transFun func(string) string) chan []string {
	out := make(chan []string)
	go func() {
		counter := counter.NewCounter()
		for _, v := range values {
			v = transFun(v)
			if counter.Has(v) {
				continue
			}
			counter.Update(v)
			// Tokenize
			tokens := tokenFun(v)
			if len(tokens) > 5 {
				// Skip text values
				continue
			}
			for i, t := range tokens {
				tokens[i] = transFun(t)
			}
			out <- tokens
		}
		close(out)
	}()
	return out
}

func add(dst, src []float64) {
	if len(dst) != len(src) {
		panic("Length of vectors not equal")
	}
	for i := range src {
		dst[i] = dst[i] + src[i]
	}
}

func VecToBytes(vec []float64, order binary.ByteOrder) []byte {
	buf := new(bytes.Buffer)
	for _, v := range vec {
		binary.Write(buf, order, v)
	}
	return buf.Bytes()
}

func BytesToVec(data []byte, order binary.ByteOrder) ([]float64, error) {
	size := len(data) / 8
	vec := make([]float64, size)
	buf := bytes.NewReader(data)
	var v float64
	for i := range vec {
		if err := binary.Read(buf, order, &v); err != nil {
			return nil, err
		}
		vec[i] = v
	}
	return vec, nil
}
