package dataflow

import "sync"

type SchedulableMessage struct {
	State  State
	Signal *sync.Cond
	End    bool
}
