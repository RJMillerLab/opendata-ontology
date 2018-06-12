package old_organization

import (
	"bufio"
	"encoding/json"
	"io/ioutil"
	"log"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
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

func CopyMap(m map[string]bool) map[string]bool {
	cm := make(map[string]bool)
	for k, v := range m {
		cm[k] = v
	}
	return cm
}

func Cosine(x, y []float64) float64 {
	if len(x) != len(y) {
		log.Printf("%d vs %d", len(x), len(y))
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

func Max(arr []float64) float64 {
	max := arr[0]
	for _, value := range arr {
		if max < value {
			max = value
		}
	}
	return max
}

func SliceToMap(arr []string) map[string]bool {
	m := make(map[string]bool)
	for _, e := range arr {
		m[e] = true
	}
	return m
}

func flatten2DSlide(s [][]float64) []float64 {
	f := make([]float64, 0)
	for _, r := range s {
		f = append(f, r...)
	}
	return f
}

func stringSlideToFloat(s [][]string) [][]float64 {
	fs := make([][]float64, 0)
	for _, r := range s {
		rf := make([]float64, 0)
		for _, e := range r {
			if f, err := strconv.ParseFloat(e, 64); err == nil {
				rf = append(rf, f)
			} else {
				panic(err)
			}
		}
		fs = append(fs, rf)
	}
	return fs
}

func getNormalDKL(s1, s2 [][]float64) float64 {
	eps := 0.0001
	//s1 := [][]float64{[]float64{0.1, 0.2, 0.4}, []float64{0.1, 0.4, 0.1}}
	//s2 := [][]float64{[]float64{0.1, 0.2, 0.1}, []float64{0.5, 0.4, 0.1}}
	dim := len(s1[0])
	fs1 := flatten2DSlide(s1)
	fs2 := flatten2DSlide(s2)
	p := mat.NewDense(len(s1), dim, fs1)
	q := mat.NewDense(len(s2), dim, fs2)
	pv := stat.CovarianceMatrix(nil, p, nil)
	qv := stat.CovarianceMatrix(nil, q, nil)
	dpv := mat.Det(pv)
	dqv := mat.Det(qv)
	pm := mean(s1)
	qm := mean(s2)
	mdiff := mat.NewDense(1, dim, nil)
	mdiff.Sub(mat.NewDense(1, dim, qm), mat.NewDense(1, dim, pm))
	qvi := mat.NewDense(0, 0, nil)
	qvi.Inverse(qv)
	iqvMpv := mat.NewDense(dim, dim, nil)
	iqvMpv.Mul(qvi, pv)
	diffMiqv := mat.NewDense(1, dim, nil)
	diffMiqv.Mul(mdiff, qvi)
	diffMiqvMdiff := mat.NewDense(1, 1, nil)
	diffMiqvMdiff.Mul(diffMiqv, mdiff.T())
	log.Printf("%f  %f ", math.Abs(dpv), math.Abs(dqv))
	return 0.5 + mat.Sum(diffMiqvMdiff) + mat.Sum(iqvMpv) + math.Log((math.Abs(dpv)+eps)/(math.Abs(dqv)+eps)) - float64(dim)
}

func hasNan(m *mat.Dense) bool {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if math.IsNaN(m.At(i, j)) || math.IsInf(m.At(i, j), 0) {
				return true
			}
		}
	}
	return false
}

func mean(d [][]float64) []float64 {
	m := make([]float64, len(d[0]))
	for _, r := range d {
		for i, v := range r {
			m[i] += v
		}
	}
	for i, _ := range m {
		m[i] = m[i] / float64(len(d))
	}
	return m
}

func (s1 state) equalState(s2 state) bool {
	for l, _ := range s1.labels {
		if _, ok := s2.labels[l]; !ok {
			return false
		}
	}
	return true
}

func (p1 *path) equalPath(p2 path) bool {
	if len(p1.states) != len(p2.states) {
		return false
	}
	for i, s1 := range p1.states {
		if s1.equalState(p2.states[i]) == false {
			return false
		}
	}
	return true
}

func avg(vecs [][]float64) []float64 {
	s := make([]float64, len(vecs[0]))
	a := make([]float64, len(vecs[0]))
	for _, vec := range vecs {
		for j, v := range vec {
			s[j] += v
		}
	}
	for j, _ := range s {
		a[j] = s[j] / float64(len(vecs))
	}
	return a
}

func expectedValue(xps []float64, p float64) float64 {
	e := 0.0
	for _, v := range xps {
		e += v
	}
	return e * p
}

func dumpJson(file string, v interface{}) error {
	buffer, err := json.Marshal(v)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(file, buffer, 0664)
	if err != nil {
		return err
	}
	return nil
}
