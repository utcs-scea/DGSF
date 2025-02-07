package models

import (
	fspb "alouatta/pb/functionserver"
	"context"
	"sync"

	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
)

// Node contains the ip address of the node
type Node struct {
	sync.RWMutex
	HostIP    string
	NodeID    string
	Conn      *grpc.ClientConn
	Grpc      fspb.FunctionClient
	TimesUsed uint32
	//dont trust TotalCreated unless you are in loadbalance.go, since it's the only thing that updates this number
	TotalCreated uint32
}

// RequestCreateReplica send the CreateRequest rpc and returns the replied ips
//this is sent to functionserver
func (n *Node) RequestCreateReplica(req *fspb.CreateRequest) (*fspb.CreateReply, error) {
	ctx := context.Background()
	vm, err := n.Grpc.CreateReplica(ctx, req)
	if err != nil {
		log.Fatal().Err(err).Msgf("%v.CreateReplica(_) = _", n.Grpc)
	}

	return vm, nil
}
