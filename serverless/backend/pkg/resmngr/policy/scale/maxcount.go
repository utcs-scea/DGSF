package scale

import (
	"github.com/rs/zerolog/log"
	"sync"
)

// MaxCountPolicy ..
type MaxCountPolicy struct {
	sync.RWMutex

	maxCount     uint32
	currentCount uint32
	reuse        bool
	deployMin    bool
	noVMScaleUp  bool
}

//NewMaxCountPolicy  ..
func NewMaxCountPolicy(cfg map[interface{}]interface{}) *MaxCountPolicy {
	return &MaxCountPolicy{
		maxCount:     uint32(cfg["max"].(int)),
		reuse:        cfg["reuse"].(bool),
		deployMin:    cfg["deploy_min"].(bool),
		noVMScaleUp:  cfg["no_vm_scaleup"].(bool),
		currentCount: 0,
	}
}

//OnFunctionDeploy ..
func (p *MaxCountPolicy) OnFunctionDeploy(minReplicas int, maxReplicas int) uint32 {
	//we dont create any on deploy, so just pass
	if p.deployMin {
		p.Lock()
		defer p.Unlock()
		p.currentCount += uint32(minReplicas)
		return uint32(minReplicas)
	}
	return 0
}

//OnRequestNoneAvailable ..
func (p *MaxCountPolicy) OnRequestNoneAvailable() uint32 {
	//we create one, up to maxcount
	if p.currentCount >= p.maxCount || p.noVMScaleUp {
		log.Debug().Msg("NO MORE VMS. MaxCount reached max")
		return 0
	}

	//else we scale 1
	p.Lock()
	defer p.Unlock()
	p.currentCount++
	return 1
}

//OnFunctionCompletion ..
func (p *MaxCountPolicy) OnFunctionCompletion() (bool, uint32) {
	//decrease our counter, return kill
	if p.reuse {
		return false, 0
	}
	p.Lock()
	defer p.Unlock()
	p.currentCount--
	return true, 0
}
