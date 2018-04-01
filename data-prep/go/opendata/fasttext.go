package opendata

import (
	"database/sql"
	"fmt"
	"math"

	fasttext "github.com/ekzhu/go-fasttext"
	"gonum.org/v1/gonum/mat"
)

type FastText struct {
	db       *sql.DB
	tokenFun func(string) []string
	transFun func(string) string
}

// Creates an in-memory FastText using an existing on-disk FastText Sqlite3 database.
func InitFastText(dbFilename string, tokenFun func(string) []string, transFun func(string) string) (*FastText, error) {
	db, err := sql.Open("sqlite3", dbFilename+"?cache=shared")
	if err != nil {
		return nil, err
	}
	return &FastText{
		db:       db,
		tokenFun: tokenFun,
		transFun: transFun,
	}, nil
}

// Creates an in-memory FastText using an existing on-disk FastText Sqlite3 database.
func InitInMemoryFastText(dbFilename string, tokenFun func(string) []string, transFun func(string) string) (*FastText, error) {
	db, err := sql.Open("sqlite3", "file::memory:?cache=shared")
	_, err = db.Exec(fmt.Sprintf(`
	attach database '%s' as disk;
	`, dbFilename))
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`create table fasttext as select * from disk.fasttext;`)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`create index inx_ft on fasttext(word);`)
	if err != nil {
		return nil, err
	}
	return &FastText{
		db:       db,
		tokenFun: tokenFun,
		transFun: transFun,
	}, nil
}

// Alaways close the FastText after finishing using it.
func (ft *FastText) Close() error {
	return ft.db.Close()
}

// Get all words that exist in the database
func (ft *FastText) GetAllWords() ([]string, error) {
	var count int
	if err := ft.db.QueryRow(`SELECT count(word) FROM fasttext;`).Scan(&count); err != nil {
		return nil, err
	}
	words := make([]string, 0, count)
	rows, err := ft.db.Query(`SELECT word FROM fasttext;`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var word string
		if err := rows.Scan(&word); err != nil {
			return words, err
		}
		words = append(words, word)
	}
	return words, rows.Err()
}

// Get the embedding vector of a word
func (ft *FastText) GetEmb(word string) ([]float64, error) {
	var binVec []byte
	err := ft.db.QueryRow(`SELECT emb FROM fasttext WHERE word=?;`, word).Scan(&binVec)
	if err == sql.ErrNoRows {
		return nil, ErrNoEmbFound
	}
	if err != nil {
		panic(err)
	}
	vec, err := BytesToVec(binVec, fasttext.ByteOrder)
	return vec, err
}

// Get the embedding vector of a data value, which is the sum of word embeddings
func (ft *FastText) GetValueEmb(value string) ([]float64, error) {
	tokens := Tokenize(value, ft.tokenFun, ft.transFun)
	return ft.getTokenizedValueEmb(tokens)
}

func (ft *FastText) GetValueEmbStrict(value string) ([]float64, error) {
	tokens := Tokenize(value, ft.tokenFun, ft.transFun)
	return ft.getTokenizedValueEmbStrict(tokens)
}

// Returns the domain embedding by summation given the
// distinct values and their frequencies
func (ft *FastText) GetDomainEmbSum(values []string, freqs []int) ([]float64, error) {
	var sum []float64
	for i, value := range values {
		freq := freqs[i]
		tokens := Tokenize(value, ft.tokenFun, ft.transFun)
		vec, err := ft.getTokenizedValueEmb(tokens)
		if err != nil {
			continue
		}
		for j, x := range vec {
			vec[j] = x * float64(freq)
		}
		if sum == nil {
			sum = vec
		} else {
			add(sum, vec)
		}
	}
	if sum == nil {
		return nil, ErrNoEmbFound
	}
	return sum, nil
}

// Returns the mean of domain embedding matrix
func (ft *FastText) GetDomainEmbMean(values []string, freqs []int) ([]float64, int, error) {
	var sum []float64
	ftValuesNum := 0
	for i, value := range values {
		freq := freqs[i]
		tokens := Tokenize(value, ft.tokenFun, ft.transFun)
		vec, err := ft.getTokenizedValueEmb(tokens)
		if err != nil {
			continue
		}
		ftValuesNum += freq
		for j, x := range vec {
			vec[j] = x * float64(freq)
		}
		if sum == nil {
			sum = vec
		} else {
			add(sum, vec)
		}
	}
	if sum == nil {
		return nil, 0, ErrNoEmbFound
	}
	mean := multVector(sum, 1.0/float64(ftValuesNum))
	return mean, ftValuesNum, nil
}

// Returns the mean and covar of domain embedding matrix
func (ft *FastText) GetDomainEmbMeanVar(values []string, freqs []int) ([]float64, []float64, int, error) {
	dim := 300
	sum := make([]float64, dim)
	mean := make([]float64, dim)
	covarSum := make([]float64, dim)
	covar := make([]float64, dim)
	ftValuesNum := 0
	for i, value := range values {
		freq := freqs[i]
		tokens := Tokenize(value, ft.tokenFun, ft.transFun)
		vec, err := ft.getTokenizedValueEmb(tokens)
		if err != nil {
			continue
		}
		//vec := vecs[i]
		ftValuesNum += freq
		for j := 0; j < dim; j++ {
			sum[j] += (float64(freq) * vec[j])
			covarSum[j] += (float64(freq) * vec[j] * vec[j])
		}
	}
	for j := 0; j < dim; j++ {
		mean[j] = sum[j] / float64(ftValuesNum)
		covarSum[j] = covarSum[j] / float64(ftValuesNum)
	}

	for j := 0; j < dim; j++ {
		covar[j] = covarSum[j] - math.Pow(mean[j], 2.0)
	}
	return mean, covar, ftValuesNum, nil
}

// Returns the mean and covar of domain embedding matrix
func (ft *FastText) GetDomainEmbMeanCovar(values []string, freqs []int) ([]float64, []float64, error) {
	dim := 300
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
	for i, value := range values {
		freq := freqs[i]
		tokens := Tokenize(value, ft.tokenFun, ft.transFun)
		vec, err := ft.getTokenizedValueEmb(tokens)
		if err != nil {
			continue
		}
		ftValuesNum += freq
		for j := 0; j < dim; j++ {
			sum[j] += (float64(freq) * vec[j])
			for k := 0; k < dim; k++ {
				covarSum[j][k] += (float64(freq) * vec[j] * vec[k])
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

func multVector(v []float64, s float64) []float64 {
	sv := make([]float64, len(v))
	for i := 0; i < len(v); i++ {
		sv[i] = v[i] / s
	}
	return sv
}

func flattenMatrix(a mat.Matrix) []float64 {
	r, _ := a.Dims()
	f := make([]float64, 0)
	for i := 0; i < r; i++ {
		f = append(f, mat.Row(nil, i, a)...)
	}
	return f
}

func flatten2DSlice(a [][]float64) []float64 {
	f := make([]float64, 0)
	for i := 0; i < len(a); i++ {
		f = append(f, a[i]...)
	}
	return f
}

// Returns the embedding vector of a tokenized data value
func (ft *FastText) getTokenizedValueEmbStrict(tokenizedValue []string) ([]float64, error) {
	var valueVec []float64
	for _, token := range tokenizedValue {
		if len(token) == 0 {
			continue
		}
		emb, err := ft.GetEmb(token)
		if err == ErrNoEmbFound {
			return nil, err
		}
		if err != nil {
			panic(err)
		}
		if valueVec == nil {
			valueVec = emb
		} else {
			add(valueVec, emb)
		}
	}
	if valueVec == nil {
		return nil, ErrNoEmbFound
	}
	return valueVec, nil
}

// Returns the embedding vector of a tokenized data value
func (ft *FastText) getTokenizedValueEmb(tokenizedValue []string) ([]float64, error) {
	var valueVec []float64
	for _, token := range tokenizedValue {
		emb, err := ft.GetEmb(token)
		if err == ErrNoEmbFound {
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
	}
	if valueVec == nil {
		return nil, ErrNoEmbFound
	}
	return valueVec, nil
}

// Tokenize the value v with tokenization and transformation function
func Tokenize(v string, tokenFun func(string) []string, transFun func(string) string) []string {
	v = transFun(v)
	tokens := tokenFun(v)
	for i, t := range tokens {
		tokens[i] = transFun(t)
	}
	return tokens
}
