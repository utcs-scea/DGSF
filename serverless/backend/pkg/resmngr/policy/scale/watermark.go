package scale

import (
	"sync"
)

// WatermarkPolicy ..
type WatermarkPolicy struct {
	high    uint32
	low     uint32
	current uint32
	reuse   bool
	sync.RWMutex
}

//NewWatermarkPolicy  ..
func NewWatermarkPolicy(cfg map[interface{}]interface{}) *WatermarkPolicy {
	return &WatermarkPolicy{
		high:    uint32(cfg["high"].(int)),
		low:     uint32(cfg["low"].(int)),
		reuse:   cfg["reuse"].(bool),
		current: 0,
	}
}

//OnFunctionDeploy ..
func (p *WatermarkPolicy) OnFunctionDeploy(minReplicas int, maxReplicas int) uint32 {
	p.Lock()
	defer p.Unlock()
	//we start at high
	p.current = p.high
	return p.high
}

//OnRequestNoneAvailable ..
func (p *WatermarkPolicy) OnRequestNoneAvailable() uint32 {
	//we create one, up to high
	if p.current >= p.high {
		return 0
	}

	//else we scale 1
	p.Lock()
	defer p.Unlock()
	p.current++
	return 1
}

//OnFunctionCompletion ..
func (p *WatermarkPolicy) OnFunctionCompletion() (bool, uint32) {
	//decrease our counter, return kill
	p.Lock()
	defer p.Unlock()
	p.current--

	//default is kill. if reuse is true, let alive
	kill := true
	if p.reuse {
		kill = false
	}

	//if less than low, go back to high
	if p.current < p.low {
		empty := p.high - p.current
		p.current = p.high
		return kill, empty
	}

	return kill, 0
}
