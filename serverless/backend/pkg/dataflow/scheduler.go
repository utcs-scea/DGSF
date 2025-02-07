package dataflow

import (
	rm "alouatta/pb/resmngr"
	"alouatta/pkg/models"
	"alouatta/pkg/resmngr"
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"

	"github.com/gofrs/uuid"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
)

//Scheduler defines functions that a scheduler need to implement
type Scheduler interface {
	//the functions that need overloading are Schedule and RunScheduler
	RunScheduler(schedulerChan <-chan SchedulableMessage) error
	ChooseReplica(state *TaskState) (string, error) //TODO: make hint here
	//these are implemented in BaseScheduler and don't need to be overloaded
	Schedule(state *TaskState) error
	RunSingleFunction(name, input string,
		schedulerHints *SchedulerHint, dfid string, nested bool) (InterStateMessage, error)
	RunDataflow(name, input string) (string, error)
	RegisterDataFlow(name string, rawDataflow []byte) error
	GetDataFlow(name string) *Dataflow
	SetProxyClient(pc *http.Client)
	CopyResources(Scheduler)
	GetBaseScheduler() *BaseScheduler
}

type GPUScheduler interface {
	ChooseWorker(state *TaskState) (*models.GPUNode, error)
}

//BaseScheduler ..
type BaseScheduler struct {
	conn           *grpc.ClientConn
	grpc           rm.ResMngrServiceClient
	proxyClient    *http.Client
	dflock         sync.RWMutex
	dataflows      map[string]*Dataflow
	rescStates     *resmngr.ResourceManagerStates
	ChildScheduler Scheduler
	GPUScheduler   GPUScheduler
	Scheduler
}

//NewBaseScheduler ..
func NewBaseScheduler(rescStates *resmngr.ResourceManagerStates) BaseScheduler {
	resmngrAddr := fmt.Sprintf("%s:%s", os.Getenv("RESMNGR_ADDR"), os.Getenv("RESMNGR_PORT"))
	log.Warn().Msgf("dialing to resmngrAddr at %v", resmngrAddr)

	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())
	conn, err := grpc.Dial(resmngrAddr, opts...)
	if err != nil {
		log.Fatal().Err(err).Msg("fail to dial")
	}
	client := rm.NewResMngrServiceClient(conn)

	return BaseScheduler{
		conn:       conn,
		grpc:       client,
		dataflows:  make(map[string]*Dataflow),
		rescStates: rescStates,
	}
}

//GetBaseScheduler ..
func (s *BaseScheduler) GetBaseScheduler() *BaseScheduler {
	return s
}

//GetRescStates returns resource manager states
func (s *BaseScheduler) GetRescStates() *resmngr.ResourceManagerStates {
	return s.rescStates
}

//RegisterDataFlow ..
func (s *BaseScheduler) RegisterDataFlow(name string, rawDataflow []byte) error {
	df, err := ParseJSONFromBytes(rawDataflow)
	if err != nil {
		return err
	}

	s.dflock.Lock()
	defer s.dflock.Unlock()
	s.dataflows[name] = df
	return nil
}

//GetDataFlow ..
func (s *BaseScheduler) GetDataFlow(name string) *Dataflow {
	s.dflock.RLock()
	defer s.dflock.RUnlock()
	return s.dataflows[name]
}

//SetProxyClient ..
func (s *BaseScheduler) SetProxyClient(pc *http.Client) {
	s.proxyClient = pc
}

//RunDataflow ..
func (s *BaseScheduler) RunDataflow(name, inputData string) (string, error) {
	df := s.GetDataFlow(name)
	if df == nil {
		log.Error().Msgf("DataFlow is not registered: %v", name)
		return "", fmt.Errorf("DataFlow is not registered: %v", name)
	}

	schedulerChan := make(chan SchedulableMessage)
	go s.ChildScheduler.RunScheduler(schedulerChan)

	dfid, err := uuid.NewV4()
	if err != nil {
		log.Error().Msg("task cant generate UUID v4")
	}
	output := df.Execute(inputData, schedulerChan, dfid.String(), 0)
	//tell the scheduler it's over
	schedulerChan <- SchedulableMessage{End: true}

	return output, nil
}

//RunSingleFunction ..
func (s *BaseScheduler) RunSingleFunction(name, inputData string,
	schedulerHints *SchedulerHint, dfid string, nested bool) (InterStateMessage, error) {
	//create state and launch async
	state := NewTaskState(name, name, "", "", "", "", true, true, false)
	state.SchedulingHints = schedulerHints
	state.Nested = nested
	state.GetBaseState().DataFlowId = dfid
	s.ChildScheduler.Schedule(state)

	//create/get recv/send channels
	input := state.ConnectInitial()
	output := state.ConnectLast()

	go state.Execute()

	//write input, which will trigger execution
	input <- InterStateMessage{Body: inputData}
	//wait for output
	respMsg := <-output

	return respMsg, nil
}

//Schedule choose where to execute something and return an IP:port
func (s *BaseScheduler) Schedule(task *TaskState) error {
	fnName := task.GetFunctionName()
	notified := false
	var err error

	//TODO: improve this function by modularizing it
	//TODO: for now we make each function wait for a GPU, then it can get a VM
	gpunode, err := s.GPUScheduler.ChooseWorker(task)
	if err != nil {
		log.Error().Msgf("Error while choosing GPU for fn: %s", err)
		return fmt.Errorf("Error while choosing GPU for fn: %s", err)
	}

	//it might have asked for a gpu
	if gpunode != nil {
		task.GPUNode = gpunode
	}

	var vmid string
	//loop until we can schedule this function
	for {
		vmid, err = s.ChildScheduler.ChooseReplica(task)
		//we found one, we break out
		if err == nil {
			break
		}

		//if we cant schedule and havent notified resmngr, do it
		if !notified {
			log.Debug().
				Str("fnName", fnName).
				Str("err", err.Error()).
				Msg("couldnt find idle vm, asking resmngr to create one")

			//TODO: better affinity
			req := &resmngr.ResourceRequest{
				Function: task.GetFunctionName(),
				Affinity: "",
				Amount:   1,
			}
			ctx := context.Background()
			//this grpc only returns when there is at least one VM available (I hope)
			err = s.rescStates.RequestVM(ctx, req)
			if err != nil {
				log.Info().Err(err).Msg("Still going to loop to retry scheduling")
			}
			nextState := task.GetBaseState().NextStateIntr
			for nextState != nil {
				t := nextState.GetType()
				if t == "Task" {
					ntask := nextState.(*TaskState)
					nreq := &resmngr.ResourceRequest{
						Function: ntask.GetFunctionName(),
						Affinity: "",
						Amount:   1,
					}
					nctx := context.Background()
					_ = s.rescStates.RequestVM(nctx, nreq)
				}
				nextState = nextState.GetBaseState().NextStateIntr
			}
			notified = true
		}
		//lets be nice and sleep between 5 and 20 ms
		sleep := time.Duration(rand.Intn(20-5) + 5)
		time.Sleep(sleep * time.Millisecond)
	}

	s.rescStates.Resources.ReplicasLock.RLock()
	urlStr := fmt.Sprintf("http://%s:%s", s.rescStates.Resources.Replicas[vmid].DeployNode.HostIP, os.Getenv("FUNCTIONPROXY_PORT"))
	s.rescStates.Resources.ReplicasLock.RUnlock()
	executor, err := url.Parse(urlStr)
	if err != nil {
		log.Error().Str("sched", "random").Msgf("Parsing the VMs url failed: %s", urlStr)
		return fmt.Errorf("error parsing url %v %v", urlStr, err)
	}

	task.ExecutorAddress = executor
	s.rescStates.Resources.ReplicasLock.RLock()
	task.FunctionIP = s.rescStates.Resources.Replicas[vmid].IP
	task.Vmid = vmid
	s.rescStates.Resources.ReplicasLock.RUnlock()
	task.ProxyClient = s.proxyClient
	task.sch = s

	return nil
}
