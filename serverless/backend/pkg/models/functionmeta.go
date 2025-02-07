package models

import (
	fspb "alouatta/pb/functionserver"
	fnserver "alouatta/pkg/functionserver"
	"sync"

	"github.com/rs/zerolog/log"
)

// FunctionMeta stores metadata for this serverless function
type FunctionMeta struct {
	Name                      string
	Replicas                  map[string]*Replica
	ExecEnvironmentPerReplica int
	//Cond variable to signal if someone is waiting for a VM to be created
	//CondLock     *sync.Mutex
	//WaitingForVM *sync.Cond
	//below are fields required to call the fcdaemon grpc
	Image       string
	EnvProcess  string
	Labels      *map[string]string
	Annotations *map[string]string
	EnvVars     map[string]string
	GPUMemReq   int64
	sync.RWMutex
}

// GetMinReplicaCount gets the minimum replica count for this function
func (m *FunctionMeta) GetMinReplicaCount() int {
	return fnserver.GetMinReplicaCount(*m.Labels)
}

// GetMaxReplicaCount gets the maximum replica count for this function
func (m *FunctionMeta) GetMaxReplicaCount() int {
	return fnserver.GetMaxReplicaCount(*m.Labels)
}

func (m *FunctionMeta) scaleUpRPC(node *Node, c chan *fspb.CreateReply) {
	req, err := fnserver.PrepareReplicaRequest(*m.Labels, m.EnvVars, m.Image)
	if err != nil {
		log.Fatal().Err(err).Msg("fail to prepare replica")
	}

	//log.Printf("ScaleUp: Sending RPC request to Node (%v)", node.HostIP)

	vm, err := node.RequestCreateReplica(req)
	if err != nil {
		log.Fatal().Err(err).Msg("fail to create replica")
	}

	c <- vm
}

// ScaleUp scales up n copy of this function on node
func (m *FunctionMeta) ScaleUp(node *Node, n uint32, rc chan *Replica) {
	c := make(chan *fspb.CreateReply)

	//create n goroutines to spawn and wait for VMs
	for i := uint32(0); i < n; i++ {
		go m.scaleUpRPC(node, c)
	}

	//receive replies
	for i := uint32(0); i < n; i++ {
		//read reply from channel and create Replica
		reply := <-c
		rep := Replica{
			Vmid:              reply.Vmid,
			IP:                reply.Ip,
			Function:          m.Name,
			Status:            VMCreating,
			DeployNode:        node,
			IdleExecutionEnvs: 1,
		}

		//send to whoever called us, so they can update the replicas map
		rc <- &rep
	}
	close(c)
}
