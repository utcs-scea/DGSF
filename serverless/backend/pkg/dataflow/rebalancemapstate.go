package dataflow

import (
	"sort"
	"strings"

	"github.com/Jeffail/gabs/v2"
	"github.com/gofrs/uuid"
	"github.com/rs/zerolog/log"
)

//MapState ..
type RebalanceMapState struct {
	dfs           []*Dataflow
	itemsPath     string
	inputPath     string
	resultPath    string
	workItemsPath string
	unitWorkPath  string
	BaseState
}

//NewMapState ..
func NewRebalanceMapState(name string, json *gabs.Container, next string, isEnd bool) *RebalanceMapState {
	children := json.ChildrenMap()
	parts, ok := children["Parts"]
	if !ok {
		log.Error().Msg("map state does not have Parts required field")
	}

	var items, ipath, rpath, wipath, uwpath string
	if items, ok = children["ItemsPath"].Data().(string); !ok {
		items = ""
	}
	if ipath, ok = children["InputPath"].Data().(string); !ok {
		ipath = ""
	}
	if rpath, ok = children["ResultPath"].Data().(string); !ok {
		rpath = ""
	}
	if wipath, ok = children["WorkItemsPath"].Data().(string); !ok {
		wipath = "$.works"
	}
	if uwpath, ok = children["UnitWorkPath"].Data().(string); !ok {
		uwpath = "$.unit"
	}

	var dfs []*Dataflow
	for _, iterator := range parts.Children() {
		df, err := ParseJSON(iterator)
		if err != nil {
			log.Error().Err(err).Msg("Err parsing map state iterator")
		}
		dfs = append(dfs, df)
	}

	return &RebalanceMapState{
		BaseState: BaseState{
			Name:      name,
			NextState: next,
			IsEnd:     isEnd,
		},
		dfs:           dfs,
		itemsPath:     items,
		inputPath:     ipath,
		resultPath:    rpath,
		workItemsPath: wipath,
		unitWorkPath:  uwpath,
	}
}

//GetType returns type "RebalanceMap"
func (m *RebalanceMapState) GetType() string {
	return "RebalanceMap"
}

//Clone ..
func (m *RebalanceMapState) Clone() State {
	c := *m
	return &c
}

func naiveRebalance(data *gabs.Container, workItemsPath string, unitWorkPath string) *gabs.Container {
	wipath := workItemsPath[2:]
	uwpath := unitWorkPath[2:]
	var allWorks []*gabs.Container
	totalWorkUnit := 0
	for idx, entry := range data.Children() {
		totalUnit := 0
		for _, work := range entry.Path(wipath).Children() {
			totalUnit += int(work.Path(uwpath).Data().(float64))
			allWorks = append(allWorks, work)
		}
		log.Info().Msgf("%d-th branch has %d units of work", idx, totalUnit)
		totalWorkUnit += totalUnit
	}
	if totalWorkUnit == 0 {
		return data
	}
	nWorkers := len(data.Children())
	unitPerWorker := (totalWorkUnit-1)/nWorkers + 1

	output, _ := gabs.ParseJSON([]byte(data.String()))

	i := 0
	for idx, entry := range output.Children() {
		j := i
		totalUnit := 0
		for j < len(allWorks) {
			totalUnit += int(allWorks[j].Path(uwpath).Data().(float64))
			j++
			if idx != nWorkers-1 && totalUnit >= unitPerWorker {
				break
			}
		}

		entry.Set(allWorks[i:j], wipath)
		i = j
		log.Info().Msgf("%d-th branch gets %d units of work", idx, totalUnit)
	}

	return output
}

//Execute for RebalanceMapState
//TODO: we can make this loop in the base class State
func (m *RebalanceMapState) Execute() {
	for {
		select {
		case inputMsg := <-m.In:
			m.executeMessage(inputMsg)
		case <-m.done:
			return
		}
	}
}

//Execute for RebalanceMapState
func (m *RebalanceMapState) executeMessage(inputMsg InterStateMessage) {
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

	currentInput := inputJSON
	schedulerChan := inputMsg.SchedulerChan
	var currentOutput *gabs.Container

	var dfid_arr []string

	for idx, df := range m.dfs {
		log.Debug().Msgf("Start %d-th sub-dataflow", idx)
		gather := make(chan Output)
		//here we launch one dataflow per child of the json
		cnt := 0
		var outputs []Output
		numFanOut := len(currentInput.Children())
		for _, child := range currentInput.Children() {
			log.Debug().Int("n", cnt).Str("input", child.String()).Msg("Executing Map child")
			if idx == 0 {
				dfid, err := uuid.NewV4()
				if err != nil {
					log.Error().Msg("task cant generate UUID v4")
				}
				dfidStr := dfid.String()
				dfid_arr = append(dfid_arr, dfidStr)
			}
			cur_dfid := dfid_arr[cnt]
			cnt++

			go func(input string, df *Dataflow, cnt int, dfid string) {
				//log.Debug().Str("fn input", input).Msg("")
				output := df.Execute(input, schedulerChan, dfid, numFanOut)
				//log.Debug().Msgf("output of map df.Execute %v", output)
				gather <- Output{
					idx:    cnt,
					output: output,
				}
			}(child.String(), df, cnt, cur_dfid)
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
		currentOutput, err = gabs.ParseJSON([]byte(output))
		if err != nil {
			log.Error().Err(err).Msgf("cant parse concatenation of result jsons: %v", currentOutput)
		}

		if idx != len(m.dfs)-1 {
			currentInput = naiveRebalance(currentOutput, m.workItemsPath, m.unitWorkPath)
		}
	}

	outputJSON := currentOutput

	if m.resultPath != "" {
		path := m.resultPath[2:]
		json.SetP(outputJSON, path)
		outputJSON = json
	}

	//write output
	inputMsg.Body = outputJSON.String()
	m.Out <- inputMsg
}
