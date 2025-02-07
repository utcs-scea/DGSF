package placement

import (
	"alouatta/pkg/models"
	"sync"

	"github.com/rs/zerolog/log"
)

// RoundRobin implements placement policy
type RoundRobin struct {
	sync.Mutex
	last uint32
}

//NewRoundRobin ..
func NewRoundRobin() *RoundRobin {
	return &RoundRobin{}
}

// OnScaleReplicaOnNode implements load balance placement policy
func (l *RoundRobin) OnScaleReplicaOnNode(nodes []*models.Node, fnmeta *models.FunctionMeta,
	n uint32, label string, c chan *models.Replica) uint32 {

	//not creating anything, so dont print a log message
	if n == 0 {
		return 0
	}

	log.Debug().Uint32("n", n).Int("nodes", len(nodes)).Msg("Creating vms round-robin across nodes")

	for i := uint32(0); i < n; i++ {
		l.Lock()
		node := l.last % uint32(len(nodes))
		go fnmeta.ScaleUp(nodes[node], 1, c)
		l.last++
		l.Unlock()
	}

	return n
}
