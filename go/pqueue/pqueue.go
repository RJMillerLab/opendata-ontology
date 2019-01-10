package pqueue

import (
	"fmt"
	"sync"
)

// PQType represents a priority queue ordering kind (see MAXPQ and MINPQ)
type PQType int

const (
	MAXPQ PQType = iota
	MINPQ
)

type item struct {
	value    interface{}
	priority float64
}

// PQueue is a heap priority queue data structure implementation.
// It can be whether max or min ordered and it is synchronized
// and is safe for concurrent operations.
type PQueue struct {
	sync.RWMutex
	items      []*item
	elemsCount int
	comparator func(float64, float64) bool
}

func newItem(value interface{}, priority float64) *item {
	return &item{
		value:    value,
		priority: priority,
	}
}

func (i *item) String() string {
	return fmt.Sprintf("<item value:%s priority:%f>", i.value, i.priority)
}

// NewPQueue creates a new priority queue with the provided pqtype
// ordering type
func NewPQueue(pqType PQType) *PQueue {
	var cmp func(float64, float64) bool

	if pqType == MAXPQ {
		cmp = max
	} else {
		cmp = min
	}

	items := make([]*item, 1)
	items[0] = nil // Heap queue first element should always be nil

	return &PQueue{
		items:      items,
		elemsCount: 0,
		comparator: cmp,
	}
}

// deletes the value item from the priority queue.
//func (pq *PQueue) Delete(value interface{}) {
//	ivalue := -1
//	log.Printf("len(pq.items): %d", len(pq.items))
//	for i, _ := range pq.items {
//		var t *item = pq.items[i]
//		log.Printf("item")
//		log.Println(t)
//		if t.value == value {
//			ivalue = i
//			break
//		}
//	}
//	pq.Lock()
//	defer pq.Unlock()

//	pq.items = append(pq.items[:ivalue], pq.items[ivalue+1:]...)
//	pq.elemsCount -= 1
//}

// Push the value item into the priority queue with provided priority.
func (pq *PQueue) Push(value interface{}, priority float64) {
	item := newItem(value, priority)

	pq.Lock()
	pq.items = append(pq.items, item)
	pq.elemsCount += 1
	pq.swim(pq.size())
	pq.Unlock()
}

// Pop and returns the highest/lowest priority item (depending on whether
// you're using a MINPQ or MAXPQ) from the priority queue
func (pq *PQueue) Pop() (interface{}, float64) {
	pq.Lock()
	defer pq.Unlock()

	if pq.size() < 1 {
		return nil, 0
	}

	var max *item = pq.items[1]

	pq.exch(1, pq.size())
	pq.items = pq.items[0:pq.size()]
	pq.elemsCount -= 1
	pq.sink(1)

	return max.value, max.priority
}

// Head returns the highest/lowest priority item (depending on whether
// you're using a MINPQ or MAXPQ) from the priority queue
func (pq *PQueue) Head() (interface{}, float64) {
	pq.RLock()
	defer pq.RUnlock()

	if pq.size() < 1 {
		return nil, 0
	}

	headValue := pq.items[1].value
	headPriority := pq.items[1].priority

	return headValue, headPriority
}

// Size returns the elements present in the priority queue count
func (pq *PQueue) Size() int {
	pq.RLock()
	defer pq.RUnlock()
	return pq.size()
}

// Check queue is empty
func (pq *PQueue) Empty() bool {
	pq.RLock()
	defer pq.RUnlock()
	return pq.size() == 0
}

func (pq *PQueue) size() int {
	return pq.elemsCount
}

func max(i, j float64) bool {
	return i < j
}

func min(i, j float64) bool {
	return i > j
}

func (pq *PQueue) less(i, j int) bool {
	return pq.comparator(pq.items[i].priority, pq.items[j].priority)
}

func (pq *PQueue) exch(i, j int) {
	var tmpItem *item = pq.items[i]

	pq.items[i] = pq.items[j]
	pq.items[j] = tmpItem
}

func (pq *PQueue) swim(k int) {
	for k > 1 && pq.less(k/2, k) {
		pq.exch(k/2, k)
		k = k / 2
	}

}

func (pq *PQueue) sink(k int) {
	for 2*k <= pq.size() {
		var j int = 2 * k

		if j < pq.size() && pq.less(j, j+1) {
			j++
		}

		if !pq.less(k, j) {
			break
		}

		pq.exch(k, j)
		k = j
	}
}
