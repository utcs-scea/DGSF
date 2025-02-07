package placement

import (
	"alouatta/pkg/models"
	"math/rand"
	"time"
)

// Random implements placement policy
type Random struct{}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// OnScaleReplicaOnNode implements load balance placement policy
func (l *Random) OnScaleReplicaOnNode(nodes []*models.Node, fnmeta *models.FunctionMeta,
	n uint32, label string, c chan *models.Replica) uint32 {

	for i := uint32(0); i < n; i++ {
		r := rand.Intn(len(nodes))
		go fnmeta.ScaleUp(nodes[r], 1, c)
	}

	return n
}
