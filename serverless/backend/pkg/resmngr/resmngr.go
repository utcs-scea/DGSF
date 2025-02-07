package resmngr

import (
	pb "alouatta/pb/resmngr"
	fnserver "alouatta/pkg/functionserver"
	"alouatta/pkg/models"
	"alouatta/pkg/resmngr/policy/placement"
	"alouatta/pkg/resmngr/policy/scale"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/openfaas/faas-provider/types"
	"github.com/rs/zerolog/log"
)

var (
	ErrRetry = errors.New("no VMs available, retry")
)

// ResourceManagerServer  has the global state of the manager
type ResourceManagerServer struct {
	pb.UnimplementedResMngrServiceServer
	States *ResourceManagerStates
}

// ResourceManagerStates contain global state
type ResourceManagerStates struct {
	Resources       *models.Resources
	ScalePolicy     scale.Interface
	PlacementPolicy placement.Interface
}

//DeployRequest request to deploy a VM
type DeployRequest struct {
	RequestJSON []byte
}

// DeployReply reply of deploy requests
type DeployReply struct {
	Ok uint32
}

//ResourceRequest request for a VM
type ResourceRequest struct {
	Function string
	Affinity string
	Amount   uint32
}

//DeployFunction handles a deploy function request coming from the faas backend
func (states *ResourceManagerStates) DeployFunction(ctx context.Context, req *DeployRequest) (*DeployReply, error) {
	request := types.FunctionDeployment{}
	if err := json.Unmarshal(req.RequestJSON, &request); err != nil {
		log.Error().Err(err).Msg("error during unmarshal of create function request.")
		return nil, err
	}

	if _, exists := states.Resources.Functions[request.Service]; exists {
		log.Warn().Str("function", request.Service).Msg("Function already deployed, overwriting..")
	}

	minReplicas := fnserver.GetMinReplicaCount(*request.Labels)
	maxReplicas := fnserver.GetMaxReplicaCount(*request.Labels)

	log.Printf("Deploying function (%v)", request.Service)

	gpuMemReq, exists := (*request.Labels)["GPU_MEM"]
	gpuMemReqMB := int64(-1)
	if exists {
		gpuMemReqMeg, err := strconv.ParseInt(gpuMemReq, 10, 64)
		if err != nil {
			log.Fatal().
				Err(err).
				Str("GPU Mem Input", gpuMemReq).
				Msg("fail to parse gpu memory input")
		}
		gpuMemReqMB = int64(gpuMemReqMeg)
		log.Debug().
			Str("GPU Mem", gpuMemReq).
			Msg("Function requires a GPU.")
	}
	//the mutex for the cond
	//lock := sync.Mutex{}
	fn := &models.FunctionMeta{
		Name:                      request.Service,
		Replicas:                  make(map[string]*models.Replica),
		ExecEnvironmentPerReplica: 1,
		//fields for fcdaemon grpc
		Image:       request.Image,
		EnvProcess:  request.EnvProcess,
		Labels:      request.Labels,
		Annotations: request.Annotations,
		EnvVars:     request.EnvVars,
		GPUMemReq:   gpuMemReqMB,
	}

	states.Resources.FunctionsLock.Lock()
	states.Resources.Functions[request.Service] = fn
	states.Resources.FunctionsLock.Unlock()

	//tell the policy a function was deployed
	reps := states.ScalePolicy.OnFunctionDeploy(minReplicas, maxReplicas)
	go states.scaleUpFunction(fn, reps)

	return &DeployReply{Ok: 1}, nil
}

func (states *ResourceManagerStates) scaleUpFunction(fnmeta *models.FunctionMeta, n uint32) {
	//we must scale up a function, this is where we make decision of which nodes and how many
	//param n is a suggestion if it's 0, otherwise its the amount we want
	if n != 0 {
		log.Debug().
			Uint32("num", n).
			Str("fn", fnmeta.Name).
			Msg("RM: scaling up function")
	}

	c := make(chan *models.Replica)

	stime := time.Now()
	//this is not pretty, but the scale thing can create more than n, so we get as return
	n = states.PlacementPolicy.OnScaleReplicaOnNode(states.Resources.Nodes, fnmeta, n, "", c)

	if n == 0 {
		//dont waste time if not spawning anything
		close(c)
		return
	}

	//update the replica map
	for i := uint32(0); i < n; i++ {
		rep := <-c
		//log.Printf("ScaleUpFunction: from channel, adding to replica map (%v)", rep.Vmid)
		states.Resources.ReplicasLock.Lock()
		states.Resources.Replicas[rep.Vmid] = rep
		states.Resources.ReplicasLock.Unlock()
	}
	duration := time.Since(stime)
	log.Debug().Dur("tototal start time", duration).
		Uint32("num vm", n).
		Int("Rep len", len(states.Resources.Replicas)).
		Str("fn", fnmeta.Name).
		Msg("startup time measured in resource manager")
	close(c)
}

//RequestVM this function is called when the scheduler can't find a vm for a function, so we need to create one or more
//this call is non-blocking, which means that after it returns there is no guarantee a VM is available
func (states *ResourceManagerStates) RequestVM(ctx context.Context, req *ResourceRequest) error {
	fnmeta, exists := states.Resources.Functions[req.Function]
	if !exists {
		return fmt.Errorf("function (%v) is not deployed", req.Function)
	}

	log.Info().Str("name", fnmeta.Name).Msg("RM: Someone requesting new VM")

	// ask scale policy if we should wait or return "retry"
	var reps uint32
	if reps = states.ScalePolicy.OnRequestNoneAvailable(); reps == 0 {
		return errors.New("policy says we cant spawn more VMs, wait or retry")
	}

	//scale up
	states.scaleUpFunction(fnmeta, reps)

	return nil
}
