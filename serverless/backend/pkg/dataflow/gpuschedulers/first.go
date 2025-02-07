package gpuschedulers

import (
	"alouatta/pkg/dataflow"
	"alouatta/pkg/models"
	"alouatta/pkg/resmngr"
	"fmt"
	"math/rand"
	"time"

	"github.com/rs/zerolog/log"
)

type FirstGPUScheduler struct {
	Resources *resmngr.ResourceManagerStates
}

func NewFirstGPUScheduler(rescStates *resmngr.ResourceManagerStates) *FirstGPUScheduler {
	rs := &FirstGPUScheduler{
		Resources: rescStates,
	}
	return rs
}

//ChooseReplica ..
func (s *FirstGPUScheduler) ChooseWorker(task *dataflow.TaskState) (*models.GPUNode, error) {
	fnName := task.GetFunctionName()
	rescs := s.Resources.Resources
	//get function meta and lock
	fnMeta, ok := rescs.Functions[fnName]
	if !ok {
		return nil, fmt.Errorf("function not deployed")
	}

	gpumem := fnMeta.GPUMemReq
	if gpumem == -1 {
		log.Debug().Msg("Function does not require GPU")
		return nil, nil
	}

	for {
		//lock all GPUnodes until we reach a decision
		s.Resources.Resources.GPUNodesLock.Lock()
		for _, gpunode := range s.Resources.Resources.GPUNodes {
			//iterate over the nodes GPUs
			if gpunode.FreeWorkers > 0 {
				gpunode.FreeWorkers -= 1
				s.Resources.Resources.GPUNodesLock.Unlock()
				log.Debug().Str("WorkerAddress", gpunode.Address).Msg("GPU worker chosen.")
				return gpunode, nil
			}

		}

		s.Resources.Resources.GPUNodesLock.Unlock()
		//instead of returning an error, let's loop until we find one
		sleep := time.Duration(rand.Intn(90) + 10)
		time.Sleep(sleep * time.Millisecond)
	}
}
