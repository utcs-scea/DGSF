package resmngr

import (
	fpb "alouatta/pb/functionserver"
	pb "alouatta/pb/resmngr"
	"alouatta/pkg/models"
	"context"
	"fmt"
	"net"
	"os"
	"time"

	"github.com/gofrs/uuid"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/peer"
)

//UpdateStatus ..
func (s *ResourceManagerServer) UpdateStatus(ctx context.Context, req *pb.StatusUpdateRequest) (*pb.StatusUpdateReply, error) {
	vmid := req.Vmid
	var rep *models.Replica = nil
	//we gotta keep trying since we are waiting for fcdaemon to send us the grpc response, then goes through 2 channels
	for i := 0; i < 200; i++ {
		s.States.Resources.ReplicasLock.RLock()
		_, exists := s.States.Resources.Replicas[vmid]
		s.States.Resources.ReplicasLock.RUnlock()
		if !exists {
			//another ugly hack
			time.Sleep(10 * time.Millisecond)
		} else {
			s.States.Resources.ReplicasLock.RLock()
			rep = s.States.Resources.Replicas[vmid]
			s.States.Resources.ReplicasLock.RUnlock()
			break
		}
	}

	if rep == nil {
		log.Fatal().
			Str("vmid", vmid).
			Msg("cant find replica on our map after 20 retries")
	}

	//When we get a VM ready notification, we need to find the FunctionMeta of that VM and signal
	//because there might be someone waiting
	if req.Status == pb.StatusUpdateRequest_READY {
		s.States.Resources.ReplicasLock.Lock()
		defer s.States.Resources.ReplicasLock.Unlock()
		s.States.Resources.FunctionsLock.RLock()
		defer s.States.Resources.FunctionsLock.RUnlock()

		s.States.Resources.Replicas[vmid].Status = models.VMIdle

		//get the replica function's meta and add it to idle vms so someone can get it
		fnmeta := s.States.Resources.Functions[rep.Function]
		fnmeta.Lock()
		//fnmeta.IdleReplicas[vmid] = s.Resources.Replicas[vmid]
		fnmeta.Replicas[vmid] = s.States.Resources.Replicas[vmid]
		fnmeta.Unlock()

		//if it's ready, mark as idle
		log.Debug().
			Str("vmid", vmid).
			Str("ip", s.States.Resources.Replicas[vmid].IP).
			Str("fn", rep.Function).
			Int("idleFns", rep.IdleExecutionEnvs).
			Msg("VM ready, added to pool")

		//we dont need the lock  to signal according to sync's docs
		//fnmeta.WaitingForVM.Signal()
		return &pb.StatusUpdateReply{Action: pb.StatusUpdateReply_KEEP_ALIVE}, nil
	} else if req.Status == pb.StatusUpdateRequest_FINISHED {
		s.States.Resources.ReplicasLock.Lock()
		defer s.States.Resources.ReplicasLock.Unlock()
		s.States.Resources.FunctionsLock.RLock()
		defer s.States.Resources.FunctionsLock.RUnlock()

		fnmeta := s.States.Resources.Functions[rep.Function]

		//ask policy what to do
		kill, reps := s.States.ScalePolicy.OnFunctionCompletion()
		go s.States.scaleUpFunction(fnmeta, reps)

		if kill {
			log.Debug().
				Str("vmid", vmid).
				Str("ip", s.States.Resources.Replicas[vmid].IP).
				Msg("VM finished, policy is kill")
			s.States.Resources.Replicas[vmid].DeployNode.Grpc.ReleaseVMSetup(ctx,
				&fpb.ReleaseRequest{Vmid: vmid})
			delete(s.States.Resources.Replicas, vmid)
			return &pb.StatusUpdateReply{Action: pb.StatusUpdateReply_DESTROY}, nil
		}

		s.States.Resources.Replicas[vmid].Status = models.VMIdle

		fnmeta.Lock()
		fnmeta.Replicas[vmid].IdleExecutionEnvs++
		fnmeta.Unlock()

		return &pb.StatusUpdateReply{Action: pb.StatusUpdateReply_KEEP_ALIVE}, nil
	}

	log.Fatal().
		Str("vmid", vmid).
		Str("ip", s.States.Resources.Replicas[vmid].IP).
		Msg("RM: VM UNKOWN status update")
	return &pb.StatusUpdateReply{Action: pb.StatusUpdateReply_DESTROY}, nil
}

//Function ..
func (s *ResourceManagerServer) RegisterFunctionNode(ctx context.Context, req *pb.RegisterFunctionNodeRequest) (*empty.Empty, error) {
	//connect to Node gRpc
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())

	ip := fmt.Sprintf("%s:%v", req.Ip, req.Port)
	log.Info().
		Str("ip", ip).
		Msg("RM: registering node, connecting back..")
	conn, err := grpc.Dial(ip, opts...)
	if err != nil {
		log.Fatal().
			Err(err).
			Msg("fail to dial")
	}
	client := fpb.NewFunctionClient(conn)

	log.Info().Msgf("RM: connected back to node (%v), register done", ip)

	node := models.Node{
		HostIP: req.Ip,
		Grpc:   client,
		Conn:   conn,
		NodeID: req.NodeId,
	}

	s.States.Resources.NodesLock.Lock()
	defer s.States.Resources.NodesLock.Unlock()
	s.States.Resources.Nodes = append(s.States.Resources.Nodes, &node)
	return &empty.Empty{}, nil
}

func (s *ResourceManagerServer) RegisterGPUNode(ctx context.Context, req *pb.RegisterGPUNodeRequest) (*pb.RegisterGPUNodeResponse, error) {
	cpeer, _ := peer.FromContext(ctx)
	nodeAddr := cpeer.Addr.String()
	nodeIp, _, _ := net.SplitHostPort(nodeAddr)
	ip := fmt.Sprintf("%s:%v", nodeIp, os.Getenv("AVAMNGR_PORT"))

	log.Info().
		Str("ip", ip).
		Msg("RM: registering GPU node.")

	//create node struct
	newuid, _ := uuid.NewV4()
	gpunode := models.GPUNode{
		Address:     ip,
		FreeWorkers: 0,
	}

	s.States.Resources.GPUNodesLock.Lock()
	s.States.Resources.GPUNodes[newuid.String()] = &gpunode
	s.States.Resources.GPUNodesLock.Unlock()

	return &pb.RegisterGPUNodeResponse{
		Uuid: newuid.String(),
	}, nil
}

func (s *ResourceManagerServer) AddGPUWorker(ctx context.Context, req *pb.AddGPUWorkerRequest) (*pb.AddGPUWorkerResponse, error) {
	s.States.Resources.GPUNodesLock.Lock()
	s.States.Resources.GPUNodes[req.Uuid].FreeWorkers += req.Workers
	log.Info().
		Uint32("addingWorkers", req.Workers).
		Uint32("totalFreeWorkers", s.States.Resources.GPUNodes[req.Uuid].FreeWorkers).
		Msg("RM: Adding free workers to GPU Node.")
	s.States.Resources.GPUNodesLock.Unlock()

	return &pb.AddGPUWorkerResponse{}, nil
}
