package pqueue

import "testing"

func Test_TopKQueue(t *testing.T) {
	sequence := []int{3, 1, 4, 5, 8, 7, 3, 9, 6, 11}
	k := 5
	queue := NewTopKQueue(k)
	for _, v := range sequence {
		queue.Push(v, float64(v))
	}
	for !queue.Empty() {
		v, priority := queue.Pop()
		t.Log(v, priority)
	}
}

func Test_TopKQueue_Descending(t *testing.T) {
	sequence := []int{3, 1, 4, 5, 8, 7, 3, 9, 6, 11}
	k := 5
	queue := NewTopKQueue(k)
	for _, v := range sequence {
		queue.Push(v, float64(v))
	}
	_, priorities := queue.Descending()
	prev := 1000.0
	for _, p := range priorities {
		if p > prev {
			t.Error("Descending does not return the correct order")
		}
		prev = p
		t.Log(p)
	}
}
