package placement

import (
	"alouatta/pkg/models"
)

//Interface is the placement policy interface
type Interface interface {
	OnScaleReplicaOnNode([]*models.Node, *models.FunctionMeta, uint32, string, chan *models.Replica) uint32
}
