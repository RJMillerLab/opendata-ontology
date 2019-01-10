package embedding

import (
	"bytes"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"

	_ "github.com/mattn/go-sqlite3"
)

var (
	ErrNoEmbFound = errors.New("No embedding found")
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
	if err != nil {
		panic(err)
	}
	_, err = db.Exec(fmt.Sprintf(`attach database '%s' as disk;`, dbFilename))
	if err != nil {
		return nil, err
	}
	// new fasttext db
	_, err = db.Exec(`create table fasttext as select * from disk.wv;`)
	//_, err = db.Exec(`create table fasttext as select * from disk.fasttext;`)
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

// Get the embedding vector of a word
func (ft *FastText) GetEmb(word string) ([]float64, error) {
	var binVec []byte
	// new fasttext db
	err := ft.db.QueryRow(`SELECT vec FROM fasttext WHERE word=?;`, word).Scan(&binVec)
	//err := ft.db.QueryRow(`SELECT emb FROM fasttext WHERE word=?;`, word).Scan(&binVec)
	if err == sql.ErrNoRows {
		return nil, ErrNoEmbFound
	}
	if err != nil {
		fmt.Println(err)
		//panic(err)
	}
	vec, err := BytesToVec(binVec, binary.LittleEndian) //fasttext.ByteOrder)
	//for _, v := range vec {
	//	if v > 1.0 || v < -1.0 {
	//		log.Println(v)
	//	}
	//}
	return vec, err
}

// Returns the mean of domain embedding matrix
func (ft *FastText) GetPhraseEmbMean(value string) ([]float64, error) {
	tokens := Tokenize(value, ft.tokenFun, ft.transFun)
	vec, err := ft.getTokenizedValueEmb(tokens)
	if err != nil {
		return nil, err
	}
	return vec, nil
}

// Returns the mean of domain embedding matrix without considering frequency
func (ft *FastText) GetDomainEmbMeanNoFreq(values []string) ([]float64, error) {
	var sum []float64
	ftValuesNum := 0
	for _, value := range values {
		tokens := Tokenize(value, ft.tokenFun, ft.transFun)
		vec, err := ft.getTokenizedValueEmb(tokens)
		if err != nil {
			continue
		}
		ftValuesNum += 1
		for j, x := range vec {
			vec[j] = x
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
	mean := multVector(sum, 1.0/float64(ftValuesNum))
	return mean, nil
}

// Returns the mean of domain embedding matrix
func (ft *FastText) GetDomainEmbMean(values []string, freqs []int) ([]float64, float64, error) {
	var sum []float64
	var card int
	ftValuesNum := 0
	for i, value := range values {
		freq := freqs[i]
		card += freq
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
		return nil, 0.0, ErrNoEmbFound
	}
	mean := multVector(sum, 1.0/float64(ftValuesNum))
	return mean, float64(ftValuesNum) / float64(card), nil
}

func multVector(v []float64, s float64) []float64 {
	sv := make([]float64, len(v))
	for i := 0; i < len(v); i++ {
		sv[i] = v[i] * s
	}
	return sv
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
			fmt.Println(err)
			//panic(err)
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
	valueVec = multVector(valueVec, 1.0/float64(len(tokenizedValue)))
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

func add(dst, src []float64) {
	if len(dst) != len(src) {
		panic("Length of vectors not equal")
	}
	for i := range src {
		dst[i] = dst[i] + src[i]
	}
}

func BytesToVec(data []byte, order binary.ByteOrder) ([]float64, error) {
	// new fasttext
	size := len(data) / 4
	//size := len(data) / 8
	vec := make([]float64, size)
	buf := bytes.NewReader(data)
	//var v float64
	// new fasttext
	var v float32
	for i := range vec {
		if err := binary.Read(buf, order, &v); err != nil {
			return nil, err
		}
		vec[i] = float64(v)
		//vec[i] =v
	}
	return vec, nil
}
