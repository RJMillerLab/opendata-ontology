package organization

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"strconv"

	"gonum.org/v1/gonum/floats"
)

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

func containsInt(as []int, i int) bool {
	for _, v := range as {
		if v == i {
			return true
		}
	}
	return false
}

func containsStr(as []string, i string) bool {
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

func sum(a [][]float64) []float64 {
	s := make([]float64, len(a[0]))
	for _, vs := range a {
		for i, v := range vs {
			s[i] += v
		}
	}
	return s
}

func updateAvg(old []float64, size int, new [][]float64) []float64 {
	s := make([]float64, len(new[0]))
	for i, v := range old {
		s[i] = v * float64(size)
	}
	for i, _ := range s {
		for _, n := range new {
			s[i] += n[i]
		}
	}
	for i, v := range s {
		s[i] = v / float64(size+len(new))
	}
	return s
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

func diff(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("dimension mismatch.")
	}
	f := make([]float64, len(a))
	for i, _ := range a {
		f[i] = a[i] - b[i]
	}
	return f
}

func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("dimension mismatch.")
	}
	d := 0.0
	for i, _ := range a {
		d += a[i] * b[i]
	}
	return d
}

func norm(a []float64) float64 {
	d := 0.0
	for _, v := range a {
		d += v * v
	}
	return math.Sqrt(d)
}

func sort(a []float64) ([]float64, []int) {
	cUP := make([]float64, len(a))
	copy(cUP, a)
	s := cUP
	inds := make([]int, len(s))
	// ascending sort
	floats.Argsort(s, inds)
	floats.Reverse(s)
	// reverse inds
	for i, j := 0, len(inds)-1; i < j; i, j = i+1, j-1 {
		inds[i], inds[j] = inds[j], inds[i]
	}
	return s, inds
}

func intersectPlus(a map[string]bool, b []string) map[string]bool {
	bm := make(map[string]bool)
	for _, v := range b {
		bm[v] = true
	}
	intersection := make(map[string]bool)
	for k, _ := range a {
		if _, ok := bm[k]; ok {
			intersection[k] = true
		}
	}
	return intersection
}

func intersect(a, b map[string]bool) map[string]bool {
	intersection := make(map[string]bool)
	for k, _ := range a {
		if _, ok := b[k]; ok {
			intersection[k] = true
		}
	}
	return intersection
}

func haveOverlap(a, b []string) bool {
	am := make(map[string]bool)
	bm := make(map[string]bool)
	for _, v := range a {
		am[v] = true
	}
	for _, v := range b {
		bm[v] = true
	}
	for v, _ := range am {
		if _, ok := bm[v]; ok {
			return true
		}
	}
	return false
}

func mapToSlice(m map[string]bool) []string {
	s := make([]string, 0)
	for v, _ := range m {
		s = append(s, v)
	}
	return s
}
