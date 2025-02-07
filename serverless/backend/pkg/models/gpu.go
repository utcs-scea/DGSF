package models

import (
	"sync"
)

//GPUNode contains gpu info for a gpu worker machine
type GPUNode struct {
	sync.RWMutex
	FreeWorkers uint32
	Address     string
}
