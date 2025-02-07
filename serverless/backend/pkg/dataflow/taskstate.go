package dataflow

import (
	"alouatta/pkg/models"
	"alouatta/pkg/util/filelog"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Jeffail/gabs/v2"
	"github.com/gofrs/uuid"
	"github.com/rs/zerolog/log"
)

type TaskState struct {
	request        *http.Request
	Resource       string    //name of function
	sch            Scheduler // for balancing the VM
	balance        bool      // whether we should spread this task's work on more than one vm
	SkipScheduling bool
	IsReady        bool
	//these three things are what the scheduler needs to fill
	ExecutorAddress *url.URL
	FunctionIP      string
	Vmid            string
	ProxyClient     *http.Client
	SchedulingHints *SchedulerHint
	inputPath       string
	workItemsPath   string
	unitWorkPath    string
	TaskID          string
	Nested          bool
	GPUNode         *models.GPUNode
	BaseState
}

//NewTaskState ..
func NewTaskState(name, fn, inputPath, wipath, uwpath, next string, isEnd bool,
	skipScheduling bool, balance bool) *TaskState {
	fid, err := uuid.NewV4()
	if err != nil {
		log.Error().Msg("task cant generate UUID v4")
	}
	return &TaskState{
		BaseState: BaseState{
			Name:      name,
			NextState: next,
			IsEnd:     isEnd,
		},
		Resource:       fn,
		balance:        balance,
		inputPath:      inputPath,
		workItemsPath:  wipath,
		unitWorkPath:   uwpath,
		SkipScheduling: skipScheduling,
		TaskID:         fid.String(),
		Nested:         false,
	}
}

//GetType returns "Task"
func (t *TaskState) GetType() string {
	return "Task"
}

//Clone ..
func (t *TaskState) Clone() State {
	c := *t
	fid, err := uuid.NewV4()
	if err != nil {
		log.Error().Msg("task cant generate UUID v4")
	}
	c.TaskID = fid.String()
	return &c
}

func (t *TaskState) executeWithInput(inputJSON *gabs.Container, bodyJSON *gabs.Container,
	inputMsg *InterStateMessage, path string) {
	//put the inputs body in the final json that will be sent
	inputJSON.SetP(bodyJSON, "input")

	//log.Debug().Msgf("input to function: %v", inputJSON.StringIndent("", "  "))

	//send the json as body of http request
	t.buildRequest("POST", strings.NewReader(inputJSON.String()), path)

	if t.Nested {
		filelog.FileLogger.Info().Str("name", t.BaseState.Name).
			Str("cat", "PERF").Str("ph", "B").Str("pid", t.GetBaseState().DataFlowId).
			Str("tid", t.FunctionIP).Msg("")
	} else {
		filelog.FileLogger.Info().Str("name", t.BaseState.Name).
			Str("cat", "PERF").Str("ph", "B").Str("pid", t.GetBaseState().DataFlowId).
			Msg("")
	}
	response, err := t.doRequest()
	if err != nil {
		log.Error().Err(err).Msg("dataflow aborted due to error for DoRequest")
		t.Out <- getErrorMsg()
		return
	}

	defer response.Body.Close()
	//update input to next fn
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Error().Err(err).Msg("error reading resp body")
		// TODO: also propagate the error out
		t.Out <- getErrorMsg()
		return
	}
	if t.Nested {
		filelog.FileLogger.Info().Str("name", t.BaseState.Name).
			Str("cat", "PERF").Str("ph", "E").Str("pid", t.BaseState.DataFlowId).
			Str("tid", t.FunctionIP).Msg("")
	} else {
		filelog.FileLogger.Info().Str("name", t.BaseState.Name).
			Str("cat", "PERF").Str("ph", "E").Str("pid", t.BaseState.DataFlowId).
			Msg("")
	}

	//parse the output of the function
	entireResponseJSON, err := gabs.ParseJSON(body)
	if err != nil {
		log.Error().Str("json", string(body)).Msg("error parsing json from function in task state")
	}
	//log.Debug().Msgf("output of msg: %v", entireResponseJSON.StringIndent("", "  "))
	output := entireResponseJSON.Path("output").String()
	//cache_information has "uploads" and "reads", as defined in hf-serverless/src/host_cache/guest_boto3wrap_py_module/boto3cached/stats.py
	pReads, _ := entireResponseJSON.Path("cache_information.reads").Data().(float64)
	inputMsg.Hints.Reads = int(pReads)
	pUps, _ := entireResponseJSON.Path("cache_information.uploads").Data().(float64)
	inputMsg.Hints.Uploads = int(pUps)
	//update the "previous" information to us
	inputMsg.Hints.ScheduledNodeIP = t.ExecutorAddress.Hostname() //this strips port, so its only ip
	inputMsg.Hints.FunctionID = t.TaskID

	//just use input with diff body
	inputMsg.Body = trimDoubleQuotes(output)
}

func (t *TaskState) rebalance(data *gabs.Container) []*gabs.Container {
	wipath := t.workItemsPath[2:]
	uwpath := t.unitWorkPath[2:]
	totalWorkUnit := 0
	var outputs []*gabs.Container
	works := data.Path(wipath).Children()
	nWorks := len(works)

	for _, work := range data.Path(wipath).Children() {
		totalWorkUnit += int(work.Path(uwpath).Data().(float64))
	}
	if nWorks == 1 {
		outputs = append(outputs, data)
		log.Warn().Int("work unit", totalWorkUnit).Msg("Use one vm to handle")
		return outputs
	}
	if totalWorkUnit == 0 || totalWorkUnit <= 100 {
		log.Warn().Int("work unit", totalWorkUnit).Msg("Use one vm to handle")
		outputs = append(outputs, data)
		return outputs
	}

	rescStates := t.sch.GetBaseScheduler().GetRescStates()
	nWorkers := 1
	fanOut := t.GetBaseState().NumFanOut
	log.Debug().Int("fanout", fanOut).Msg("")

	rescStates.Resources.NodesLock.RLock()
	nNodes := len(rescStates.Resources.Nodes)
	rescStates.Resources.NodesLock.RUnlock()

	if t.GetBaseState().NumFanOut != 0 {
		nWorkers = fanOut / nNodes
	} else {
		rescStates.Resources.FunctionsLock.RLock()
		fnmeta := rescStates.Resources.Functions[t.GetFunctionName()]
		nWorkers = len(fnmeta.Replicas) / nNodes
		rescStates.Resources.FunctionsLock.RUnlock()
	}
	unitPerWorker := (totalWorkUnit-1)/nWorkers + 1

	log.Warn().Int("nWorkers", nWorkers).Int("total work unit", totalWorkUnit).
		Int("nworks", nWorks).Int("unit per worker", unitPerWorker).
		Str("df id", t.GetBaseState().DataFlowId).Msg("")

	totalAssignedUnit := 0
	j := 0
	output, _ := gabs.ParseJSON([]byte(data.String()))
	cleaned := cleanupJSON(output)
	for i := 0; i < nWorkers; i++ {
		if i == 0 {
			output, _ = gabs.ParseJSON([]byte(data.String()))
		} else {
			output, _ = gabs.ParseJSON([]byte(cleaned.String()))
		}
		k := j
		totalUnit := 0
		for k < nWorks {
			totalUnit += int(works[k].Path(uwpath).Data().(float64))
			k++
			if i != nWorkers-1 && totalUnit >= unitPerWorker {
				break
			}
		}

		if totalUnit > 0 {
			output.Set(works[j:k], wipath)
			log.Warn().Msgf("%d-th branch gets %d units of work", i, totalUnit)
			outputs = append(outputs, output)
			totalAssignedUnit += totalUnit
		}
		j = k
	}
	return outputs
}

//Execute ..
//TODO: we can make this loop in the base class State
func (t *TaskState) Execute() {
	for {
		select {
		case inputMsg := <-t.In:
			//log.Debug().Msgf("Task state %v got a message, executing", t.GetBaseState().Name)
			t.executeMessage(inputMsg)
		case <-t.done:
			return
		}
	}
}

func (t *TaskState) executeMessage(inputMsg InterStateMessage) {
	//if this is false it means we're a single state and were already scheduled before execution
	if !t.SkipScheduling {
		//log.Debug().Str("state", t.Name).Str("input", inputMsg.Body).Msg("scheduling function")
		//create our waiting condition
		lock := sync.Mutex{}
		signal := sync.NewCond(&lock)
		signal.L.Lock()

		//store hints in ourself
		t.SchedulingHints = &inputMsg.Hints
		//create the schedulable message with information to the scheduler
		schedMsg := SchedulableMessage{
			State:  t,
			Signal: signal,
		}
		//tell scheduler we are ready..
		inputMsg.SchedulerChan <- schedMsg

		//..then wait for it to tell us we can go
		for !t.IsReady {
			signal.Wait()
		}
		signal.L.Unlock()

		//log.Debug().Str("state", t.Name).Msg("got signal from sched to continue")
	}

	//fill in the json with function information
	inputJSON := gabs.New()
	inputJSON.SetP(inputMsg.Hints.ScheduledNodeIP, "function_info.previous_node")
	inputJSON.SetP(inputMsg.Hints.FunctionID, "function_info.previous_function_id")
	inputJSON.SetP(t.TaskID, "function_info.function_id")
	if t.GPUNode != nil && t.GPUNode.Address != "" {
		inputJSON.SetP(t.GPUNode.Address, "function_info.gpu_mngr_address")
	}

	entireIncomingJSON, err := gabs.ParseJSON([]byte(inputMsg.Body))
	if err != nil {
		log.Error().
			Str("json", inputMsg.Body).
			Err(err).
			Msg("error parsing json map input")
	}
	//assume the function input body is everything
	bodyJSON := entireIncomingJSON
	//but filter if there is inputPath
	if t.inputPath != "" {
		ipath := t.inputPath[2:]
		val := entireIncomingJSON.Path(ipath)
		if val == nil {
			log.Error().Msgf("cant find input path (%v) in input json %v", ipath, entireIncomingJSON.String())
		} else {
			bodyJSON = val
		}
	}

	//Resource is an ARN, the last string :-separated is name of function (all we need)
	path := "function/" + t.GetFunctionName()
	t.executeWithInput(inputJSON, bodyJSON, &inputMsg, path)
	t.Out <- inputMsg
}

func cleanupJSON(input *gabs.Container) *gabs.Container {
	for key := range input.ChildrenMap() {
		if strings.Contains(key, "_io_t") ||
			strings.Contains(key, "_proc_t") ||
			strings.Contains(key, "_get_n") ||
			strings.Contains(key, "_get_avg") ||
			strings.Contains(key, "_put_n") ||
			strings.Contains(key, "_put_avg") ||
			strings.Contains(key, "_list_n") ||
			strings.Contains(key, "_list_avg") ||
			strings.Contains(key, "_splts_n") ||
			strings.Contains(key, "_splts_avg") ||
			strings.Contains(key, "_dled") ||
			strings.Contains(key, "_cachedled") {
			_ = input.DeleteP(key)
		}
	}
	return input
}

//GetFunctionName ..
func (t *TaskState) GetFunctionName() string {
	strs := strings.Split(t.Resource, ":")
	fn := strs[len(strs)-1]
	return fn
}

func (t *TaskState) buildRequest(method string, body io.Reader, path string) error {
	t.ExecutorAddress.Path = path
	req, err := http.NewRequest(method, t.ExecutorAddress.String(), body)
	if err != nil {
		log.Error().Err(err).Msg("err building new request")
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Redirect-To", t.FunctionIP)
	t.request = req
	return nil
}

//DoRequest send request and return response
func (t *TaskState) doRequest() (*http.Response, error) {
	start := time.Now()
	response, err := t.ProxyClient.Do(t.request)
	seconds := time.Since(start)
	if err != nil {
		log.Error().Err(err).
			Str("url", t.request.URL.String()).
			Str("fn", t.request.Header.Get("Redirect-To")).
			Float64("time (s)", seconds.Seconds()).
			Msg("error with proxy request")
		return nil, fmt.Errorf("can't reach service for: %s", t.Name)
	}
	log.Info().Str("fn", t.Name).
		Float64("time (s)", seconds.Seconds()).
		Int("code", response.StatusCode).
		Str("fnAddr", t.FunctionIP).
		Str("vmid", t.Vmid).
		Msg("fn finished")

	return response, nil
}
