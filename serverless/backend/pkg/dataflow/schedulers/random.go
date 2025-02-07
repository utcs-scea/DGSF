package schedulers

import (
	"alouatta/pkg/dataflow"
	"alouatta/pkg/models"
	"alouatta/pkg/resmngr"
	"fmt"
	"math/rand"
	"sync/atomic"

	"github.com/rs/zerolog/log"
)

// RandomScheduler is a struct that implements random selection of VM
type RandomScheduler struct {
	dataflow.BaseScheduler
}

// NewRandomScheduler creates a new random Scheduler
func NewRandomScheduler(rescStates *resmngr.ResourceManagerStates) *RandomScheduler {
	rs := &RandomScheduler{
		BaseScheduler: dataflow.NewBaseScheduler(rescStates),
	}

	rs.ChildScheduler = rs
	return rs
}

//Choice contains one possible VM and GPU assignment
type Choice struct {
	vmid, gpuUUID string
	gpuNode       *models.GPUNode
	workerPort    uint32
	gpuIdx        int
}

//RandomChoiceWithLocks returns vmid, gpuUUID, gpu worker port, error
func (s *RandomScheduler) RandomChoiceWithLocks(fnMeta *models.FunctionMeta) (string, error) {
	fnName := fnMeta.Name
	var reps []Choice

	for vmid, rep := range fnMeta.Replicas {
		//if this replica isn't available, dont waste time
		if rep.IdleExecutionEnvs == 0 {
			continue
		}
		candidate := Choice{vmid: vmid}
		reps = append(reps, candidate)
	}

	//none available, return nothing
	if len(reps) == 0 {
		return "", fmt.Errorf("no vms on random choice, total (%v)", len(fnMeta.Replicas))
	}

	//choose random one
	chosen := reps[rand.Intn(len(reps))]
	log.Debug().Str("sched", "random").
		Int("idleReplicas", len(reps)).
		Int("totalReplicas", len(fnMeta.Replicas)).
		Str("vmid", chosen.vmid).
		Str("task", fnName).
		Msg("random VM was chosen")

	//mark it as used
	fnMeta.Replicas[chosen.vmid].IdleExecutionEnvs--
	atomic.AddUint32(&fnMeta.Replicas[chosen.vmid].DeployNode.TimesUsed, 1)

	return chosen.vmid, nil
}

//ChooseReplica ..
func (s *RandomScheduler) ChooseReplica(task *dataflow.TaskState) (string, error) {
	fnName := task.GetFunctionName()
	rescs := s.GetRescStates().Resources
	//get function meta and lock
	fnMeta, ok := rescs.Functions[fnName]
	if !ok {
		return "", fmt.Errorf("function not deployed")
	}
	fnMeta.Lock()
	defer fnMeta.Unlock()

	//if there are no replicas, return err so that Schedule can talk to resmngr
	if len(fnMeta.Replicas) == 0 {
		return "", fmt.Errorf("no vms")
	}

	//do a random choice, in different function for reusability
	return s.RandomChoiceWithLocks(fnMeta)
}

//RunScheduler ..
func (s *RandomScheduler) RunScheduler(schedulerChan <-chan dataflow.SchedulableMessage) error {
	for {
		msg := <-schedulerChan
		//if it's over let's quit
		if msg.End {
			break
		}

		//go async to schedule more than one fn at a time
		go func() {
			//we only get task states for now, so lets convert
			task := msg.State.(*dataflow.TaskState)
			//choose where the function will be executed and fill the three required fields
			s.Schedule(task)
			//tell the function it can continue
			msg.Signal.L.Lock()
			task.IsReady = true
			msg.Signal.Signal()
			msg.Signal.L.Unlock()
		}()
	}
	return nil
}
