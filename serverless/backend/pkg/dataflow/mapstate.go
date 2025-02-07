package dataflow

import (
	"sort"
	"strings"

	"github.com/Jeffail/gabs/v2"
	"github.com/gofrs/uuid"
	"github.com/rs/zerolog/log"
)

//MapState ..
type MapState struct {
	df         *Dataflow
	itemsPath  string
	inputPath  string
	resultPath string
	BaseState
}

//NewMapState ..
func NewMapState(name string, json *gabs.Container, next string, isEnd bool) *MapState {
	children := json.ChildrenMap()
	iterator, ok := children["Iterator"]
	if !ok {
		log.Error().Msg("map state does not have Iterator required field")
	}

	var items, ipath, rpath string
	if items, ok = children["ItemsPath"].Data().(string); !ok {
		items = ""
	}
	if ipath, ok = children["InputPath"].Data().(string); !ok {
		ipath = ""
	}
	if rpath, ok = children["ResultPath"].Data().(string); !ok {
		rpath = ""
	}

	df, err := ParseJSON(iterator)
	if err != nil {
		log.Error().Err(err).Msg("Err parsing map state iterator")
	}

	return &MapState{
		BaseState: BaseState{
			Name:      name,
			NextState: next,
			IsEnd:     isEnd,
		},
		df:         df,
		itemsPath:  items,
		inputPath:  ipath,
		resultPath: rpath,
	}
}

//GetType returns type "Map"
func (m *MapState) GetType() string {
	return "Map"
}

//Clone ..
func (m *MapState) Clone() State {
	c := *m
	return &c
}

//Execute for MapState
//TODO: we can make this loop in the base class State
func (m *MapState) Execute() {
	for {
		select {
		case inputMsg := <-m.In:
			log.Debug().Msgf("Map state %v got a message, executing", m.GetBaseState().Name)
			m.executeMessage(inputMsg)
		case <-m.done:
			return
		}
	}
}

func (m *MapState) executeMessage(inputMsg InterStateMessage) {
	json, err := gabs.ParseJSON([]byte(inputMsg.Body))
	if err != nil {
		log.Error().
			Str("json", inputMsg.Body).
			Err(err).
			Msg("error parsing json map input")
	}
	inputJSON := json

	//filter the input first
	if m.inputPath != "" {
		ipath := m.inputPath[2:]
		val := inputJSON.Path(ipath)
		if val == nil {
			log.Error().Msgf("cant find input path (%v) in input json %v", ipath, json.String())
		} else {
			inputJSON = val
		}
	}

	//if ItemsPath was given to the Map, we have to find it
	if m.itemsPath != "" {
		path := m.itemsPath[2:] //this removes "$."
		val := inputJSON.Path(path)
		if val == nil {
			log.Error().Msgf("cant find path (%v) in input json %v", path, json.String())
		} else {
			inputJSON = val
		}
	}

	gather := make(chan Output)
	//here we launch one dataflow per child of the json
	cnt := 0
	var outputs []Output
	numFanOut := len(inputJSON.Children())
	for _, child := range inputJSON.Children() {
		log.Debug().Int("child", cnt).Str("map state", child.String()).Msg("")
		cnt++

		go func(input string, cnt int) {
			//log.Debug().Str("fn input", input).Msg("")

			dfid, err := uuid.NewV4()
			if err != nil {
				log.Error().Msg("task cant generate UUID v4")
			}
			output := m.df.Execute(input, inputMsg.SchedulerChan, dfid.String(), numFanOut)
			//log.Debug().Msgf("output of map df.Execute %v", output)
			gather <- Output{
				idx:    cnt,
				output: output,
			}
		}(child.String(), cnt)
	}
	for i := 0; i < cnt; i++ {
		out := <-gather
		outputs = append(outputs, out)
	}
	sort.Sort(ByIdx(outputs))

	var b strings.Builder
	b.WriteString("[")
	for i := 0; i < cnt; i++ {
		//log.Debug().Msgf("Map received back from df %v", i)
		//force json to be in an array
		b.WriteString(outputs[i].output.(string))
		//dont append to last element
		if i != cnt-1 {
			b.WriteString(",")
		}
	}
	b.WriteString("]")
	output := b.String()
	outputJSON, err := gabs.ParseJSON([]byte(output))
	if err != nil {
		log.Error().Err(err).Msgf("cant parse concatenation of result jsons: %v", outputJSON)
	}

	if m.resultPath != "" {
		path := m.resultPath[2:]
		json.SetP(outputJSON, path)
		outputJSON = json
	}

	//write output
	inputMsg.Body = outputJSON.String()
	m.Out <- inputMsg
}
