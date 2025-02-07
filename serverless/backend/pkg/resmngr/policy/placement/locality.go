package placement

import "alouatta/pkg/models"

// Locality implements placement policy
type Locality struct{}

// OnScaleReplicaOnNode implements locality based placement policy
// label is the IP address of the node
func (l *Locality) OnScaleReplicaOnNode(nodes []*models.Node, fnmeta *models.FunctionMeta,
	n uint32, label string, c chan *models.Replica) uint32 {
	var nodeToPlace *models.Node
	if label != "" {
		for _, node := range nodes {
			if node.HostIP == label {
				nodeToPlace = node
				break
			}
		}
	} else {
		nodeToPlace = nodes[0]
	}
	go fnmeta.ScaleUp(nodeToPlace, n, c)

	return n
}
