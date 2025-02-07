package dataflow

import (
	"github.com/Jeffail/gabs/v2"
)

//SucceedState contains data for Parallel State
type SucceedState struct {
	BaseState
}

//NewSucceedStateState creates a new ParallelState
func NewSucceedStateState(name string, json *gabs.Container) *ParallelState {
	return &ParallelState{
		BaseState: BaseState{
			Name:  name,
			IsEnd: true,
		},
	}
}

//GetType ...
func (p *SucceedState) GetType() string {
	return "Succeed"
}

//Clone ..
func (p *SucceedState) Clone() State {
	c := *p
	return &c
}

//Execute just returns the input
//succeed doesnt need a loop since it only executes once
func (p *SucceedState) Execute() {
	inputMsg := <-p.In
	p.Out <- inputMsg
}
