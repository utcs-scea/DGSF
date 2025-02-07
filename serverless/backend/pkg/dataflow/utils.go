package dataflow

import (
	"errors"

	"github.com/Jeffail/gabs/v2"
	"github.com/rs/zerolog/log"
)

//https://docs.aws.amazon.com/step-functions/latest/dg/concepts-states.html

var (
	startAtNotFoundMsg = "StartAt not found on json"
	errStartAtNotFound = errors.New(startAtNotFoundMsg)
	ErrRetry           = errors.New("no VMs available, retry")
)

type Output struct {
	idx    int
	output interface{}
}

type ByIdx []Output

func (a ByIdx) Len() int           { return len(a) }
func (a ByIdx) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByIdx) Less(i, j int) bool { return a[i].idx < a[j].idx }

//NewTaskFromJSON this is nice but at the same time requires all parameters all states can use
func NewTaskFromJSON(key string, child *gabs.Container) State {

	//get next
	var next string
	if child.Exists("Next") {
		next = child.Search("Next").Data().(string)
	} else {
		next = ""
	}

	//get last
	var last bool
	if child.Exists("End") {
		last = child.Search("End").Data().(bool)
	} else {
		last = false
	}

	//state is an interface holding all types
	var state State
	//AWS state machine spec:
	//https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-task-state.html
	t := child.S("Type").Data().(string)
	switch t {
	case "Task":
		children := child.ChildrenMap()

		ipath, ok := children["InputPath"].Data().(string)
		if !ok {
			ipath = ""
		}
		var balance bool
		balance, ok = children["Balance"].Data().(bool)
		if !ok {
			balance = false
		}
		wipath := ""
		uwpath := ""
		if balance {
			if wipath, ok = children["WorkItemsPath"].Data().(string); !ok {
				wipath = "$.works"
			}
			if uwpath, ok = children["UnitWorkPath"].Data().(string); !ok {
				uwpath = "$.unit"
			}
		}
		state = NewTaskState(key,
			child.Search("Resource").Data().(string),
			ipath,
			wipath,
			uwpath,
			next,
			last,
			false, balance)
	case "Map":
		state = NewMapState(key, child, next, last)
	case "Parallel":
		state = NewParallelState(key, child, next, last)
	case "RebalanceMap":
		state = NewRebalanceMapState(key, child, next, last)
	case "Choice":
		state = NewChoiceState(key, child)
	case "Succeed":
		state = NewSucceedStateState(key, child)
	default:
		log.Warn().Msgf("State Type %v not implemented yet", t)
	}

	return state
}

/***************************************
 ******      Other methods      ********
 ***************************************/

func getErrorMsg() InterStateMessage {
	return InterStateMessage{Body: ""}
}

func trimDoubleQuotes(s string) string {
	if len(s) > 0 && s[0] == '"' {
		s = s[1:]
	}
	if len(s) > 0 && s[len(s)-1] == '"' {
		s = s[:len(s)-1]
	}
	return s
}

//ParseJSONFromBytes ..
func ParseJSONFromBytes(input []byte) (*Dataflow, error) {

	//parse the json using gabs (golang's suck):  https://github.com/Jeffail/gabs
	json, err := gabs.ParseJSON(input)
	if err != nil {
		log.Error().Err(err).Msg("error reading req body")
		//httputil.Errorf(w, http.StatusServiceUnavailable, "Error parsing request body\n")
	}

	return ParseJSON(json)
}

//ParseJSON ..
func ParseJSON(json *gabs.Container) (*Dataflow, error) {

	log.Debug().Str("parsing json", json.String()).Msg("")

	states := make(map[string]State)
	//iterate over the json we received, building State map
	for key, child := range json.Search("States").ChildrenMap() {
		//log.Debug().Str("adding state", key).Msg("")
		//create and add it to our map
		states[key] = NewTaskFromJSON(key, child)
	}

	initialStateName, ok := json.Path("StartAt").Data().(string)
	if !ok {
		log.Error().Msg(startAtNotFoundMsg)
		return nil, errStartAtNotFound
	}

	df := &Dataflow{
		StartState: initialStateName,
		States:     states,
	}
	return df, nil
}
