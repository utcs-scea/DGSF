package models

import (
	"sync"
)

// Resources contains all resources the resource manager manges
type Resources struct {
	NodesLock sync.RWMutex
	Nodes     []*Node //map of available nodes

	GPUNodesLock sync.RWMutex
	GPUNodes     map[string]*GPUNode //map of available gpu nodes

	ReplicasLock sync.RWMutex
	Replicas     map[string]*Replica //maps vmid to replica

	FunctionsLock sync.RWMutex
	Functions     map[string]*FunctionMeta //maps function names to function meta
}

func NewResources() *Resources {
	return &Resources{
		Replicas:  make(map[string]*Replica),
		Functions: make(map[string]*FunctionMeta),
		GPUNodes:  make(map[string]*GPUNode),
	}
}
