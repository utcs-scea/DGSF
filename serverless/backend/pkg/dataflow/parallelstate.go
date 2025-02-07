package dataflow

import (
	"sort"
	"strings"

	"github.com/Jeffail/gabs/v2"
	"github.com/gofrs/uuid"

	"github.com/rs/zerolog/log"
)

//ParallelState contains data for Parallel State
type ParallelState struct {
	dfs        []*Dataflow
	resultPath string
	BaseState
}

//NewParallelState creates a new ParallelState
func NewParallelState(name string, json *gabs.Container, next string, isEnd bool) *ParallelState {
	children := json.ChildrenMap()
	branches, ok := children["Branches"]
	if !ok {
		log.Error().Msg("parallel state does not have Branches required field")
	}

	var rpath string
	if rpath, ok = children["ResultPath"].Data().(string); !ok {
		rpath = ""
	}

	var dfs []*Dataflow
	for _, branch := range branches.Children() {
		df, err := ParseJSON(branch)
		if err != nil {
			log.Error().Err(err).Msg("Err parsing parallel state branches")
		}
		dfs = append(dfs, df)
	}

	return &ParallelState{
		BaseState: BaseState{
			Name:      name,
			NextState: next,
			IsEnd:     isEnd,
		},
		dfs:        dfs,
		resultPath: rpath,
	}
}

//GetType ...
func (p *ParallelState) GetType() string {
	return "Parallel"
}

//Clone creates a copy of ParallelState
func (p *ParallelState) Clone() State {
	c := *p
	return &c
}

//Execute for ParallelState
//TODO: we can make this loop in the base class State
func (p *ParallelState) Execute() {
	for {
		select {
		case inputMsg := <-p.In:
			p.executeMessage(inputMsg)
		case <-p.done:
			return
		}
	}
}

//Execute for ParallelState
func (p *ParallelState) executeMessage(inputMsg InterStateMessage) {
	json, err := gabs.ParseJSON([]byte(inputMsg.Body))
	if err != nil {
		log.Error().
			Str("json", inputMsg.Body).
			Err(err).
			Msg("error parsing json map input")
	}

	//The elements of the output array correspond to the branches in the same order
	//that they appear in the “Branches” array. need to sort the output according to the index
	gather := make(chan Output)
	var outputs []Output
	for idx, df := range p.dfs {
		go func(df *Dataflow, cnt int) {
			dfid, err := uuid.NewV4()
			if err != nil {
				log.Error().Msg("task cant generate UUID v4")
			}
			output := df.Execute(inputMsg.Body, inputMsg.SchedulerChan, dfid.String(), 0)
			gather <- Output{
				idx:    cnt,
				output: output,
			}
		}(df, idx)
	}

	length := len(p.dfs)
	for i := 0; i < length; i++ {
		out := <-gather
		outputs = append(outputs, out)
	}
	sort.Sort(ByIdx(outputs))

	var b strings.Builder
	b.WriteString("[")
	for i := 0; i < length; i++ {
		b.WriteString(outputs[i].output.(string))
		if i != length-1 {
			b.WriteString(",")
		}
	}
	b.WriteString("]")
	output := b.String()

	outputJSON, err := gabs.ParseJSON([]byte(output))
	if err != nil {
		log.Error().Err(err).Msg("cant parse concatenation of result jsons")
	}
	if p.resultPath != "" {
		path := p.resultPath[2:]
		json.SetP(outputJSON, path)
		outputJSON = json
	}

	inputMsg.Body = outputJSON.String()
	p.Out <- inputMsg
}
