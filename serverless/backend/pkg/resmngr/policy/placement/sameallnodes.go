package placement

import "alouatta/pkg/models"

// LoadBalance implements placement policy
type SameAllNodes struct{}

// OnScaleReplicaOnNode implements load balance placement policy
func (l *SameAllNodes) OnScaleReplicaOnNode(nodes []*models.Node, fnmeta *models.FunctionMeta,
	n uint32, label string, c chan *models.Replica) uint32 {
	for i := 0; i < len(nodes); i++ {
		go fnmeta.ScaleUp(nodes[i], n, c)
	}

	return n * uint32(len(nodes))
}
