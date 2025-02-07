package placement

import (
	"alouatta/pkg/models"
	"sync"
	"sync/atomic"

	"github.com/rs/zerolog/log"
)

type nodeCountPair struct {
	Node  *models.Node
	count int
}

// LoadBalance implements placement policy
type LoadBalance struct {
	sync.RWMutex
}

//NewLoadBalance ..
func NewLoadBalance() *LoadBalance {
	return &LoadBalance{}
}

func getMinReplicas(m map[string]*nodeCountPair) string {
	minID := ""
	//min := 0
	var minTotal uint32 = 0
	for id, pair := range m {
		log.Debug().Msgf("%v has %v", pair.Node.NodeID, pair.Node.TotalCreated)
		//if minID == "" || (pair.count <= min && pair.Node.TotalCreated < minTotal) {
		if minID == "" || pair.Node.TotalCreated < minTotal {
			minID = id
			//min = pair.count
			minTotal = pair.Node.TotalCreated
		}
	}
	return minID
}

// OnScaleReplicaOnNode implements load balance placement policy
func (l *LoadBalance) OnScaleReplicaOnNode(nodes []*models.Node, fnmeta *models.FunctionMeta,
	n uint32, label string, c chan *models.Replica) uint32 {
	l.Lock()
	defer l.Unlock()

	if n == 0 {
		return 0
	}

	//this entire thing is not efficient AT ALL
	pairs := make(map[string]*nodeCountPair)
	//add all nodes to map
	for _, node := range nodes {
		//log.Debug().Msgf("adding %v", node.NodeID)
		pairs[node.NodeID] = &nodeCountPair{Node: node, count: 0}
	}

	//count how many are deployed on each
	for _, rep := range fnmeta.Replicas {
		pairs[rep.DeployNode.NodeID].count++
	}

	for _i := 0; _i < int(n); _i++ {
		minID := getMinReplicas(pairs)
		//because ScaleUp is async, we need to inc here
		atomic.AddUint32(&pairs[minID].Node.TotalCreated, 1)
		pairs[minID].count++

		//spawn one
		go fnmeta.ScaleUp(pairs[minID].Node, 1, c)
	}

	return n
}
