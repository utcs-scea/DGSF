package functionserver

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/firecracker-microvm/firecracker-go-sdk"
	"github.com/gofrs/uuid"
	"golang.org/x/sync/errgroup"
)

const (
	// as specified in http://man7.org/linux/man-pages/man8/ip-netns.8.html
	defaultNetNSDir = "/var/run/netns"
)

var (
	ErrNotAssign = errors.New("VM setup is not assigned")
)

// VMSetupPool contains a pool of vm setup
type VMSetupPool struct {
	sync.Mutex
	IdleVMSetup     []*VMSetup
	OccupiedVMSetup map[string]*VMSetup
	ctx             context.Context
	NetName         string
}

// VMSetup contains meta data for the vm and a configured network interface using cni
type VMSetup struct {
	ReplicaMeta       *ReplicaMeta
	networkInterfaces firecracker.NetworkInterfaces
	netns             string
	clearFuncs        []func() error
}

// NewVMSetupPool creates a empty vm setup pool
func NewVMSetupPool(ctx context.Context, netname string) *VMSetupPool {
	return &VMSetupPool{
		ctx:             ctx,
		NetName:         netname,
		IdleVMSetup:     make([]*VMSetup, 0, 256),
		OccupiedVMSetup: make(map[string]*VMSetup),
	}
}

func defaultNetNSPath(vmid string) string {
	return filepath.Join(defaultNetNSDir, vmid)
}

func (p *VMSetupPool) createVMSetup() (*VMSetup, error) {
	vmid, err := uuid.NewV4()
	if err != nil {
		return nil, fmt.Errorf("fail to generate uuid: %v", err)
	}
	replicaMeta, err := newReplicaMeta(p.ctx, vmid.String())
	if err != nil {
		return nil, fmt.Errorf("fail to create replica meta: %v", err)
	}
	vmSetup := &VMSetup{
		ReplicaMeta: replicaMeta,
		networkInterfaces: firecracker.NetworkInterfaces{
			firecracker.NetworkInterface{
				CNIConfiguration: &firecracker.CNIConfiguration{
					NetworkName: p.NetName,
					IfName:      "veth0",
				},
			},
		},
	}
	vmidStr := vmid.String()
	vmSetup.netns = defaultNetNSPath(vmidStr)
	err, clearFuncs := vmSetup.networkInterfaces.SetupNetwork(p.ctx, vmidStr,
		vmSetup.netns,
		replicaMeta.logger)
	if err != nil {
		return nil, err
	}
	vmSetup.clearFuncs = clearFuncs
	return vmSetup, nil
}

// FillVMSetupPool fills in n vm setup to idle pool
func (p *VMSetupPool) FillVMSetupPool(n int) error {
	g, ctx := errgroup.WithContext(p.ctx)
	data := make(chan *VMSetup)
	for i := 0; i < n; i++ {
		g.Go(func() error {
			vmSetup, err := p.createVMSetup()
			if err != nil {
				return err
			}
			select {
			case data <- vmSetup:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		})
	}
	go func() {
		g.Wait()
		close(data)
	}()
	for d := range data {
		p.Lock()
		p.IdleVMSetup = append(p.IdleVMSetup, d)
		p.Unlock()
	}
	if err := g.Wait(); err != nil {
		return err
	}
	return nil
}

// GetVMSetup returns a vm setup
func (p *VMSetupPool) GetVMSetup() (*VMSetup, error) {
	p.Lock()
	defer p.Unlock()
	if len(p.IdleVMSetup) > 0 {
		vmSetup := p.IdleVMSetup[0]
		p.IdleVMSetup = p.IdleVMSetup[1:]
		p.OccupiedVMSetup[vmSetup.ReplicaMeta.Vmid] = vmSetup
		return vmSetup, nil
	} else {
		vmSetup, err := p.createVMSetup()
		if err != nil {
			return nil, err
		}
		p.OccupiedVMSetup[vmSetup.ReplicaMeta.Vmid] = vmSetup
		return vmSetup, nil
	}
}

// ReturnVMSetup returns the vm setup back to the idle vm setup pool
func (p *VMSetupPool) ReturnVMSetup(vmSetup *VMSetup) {
	p.Lock()
	defer p.Unlock()

	delete(p.OccupiedVMSetup, vmSetup.ReplicaMeta.Vmid)
	p.IdleVMSetup = append(p.IdleVMSetup, vmSetup)
}

// ReturnVMSetupByVmid returns the vm setup identified by vmid to the idle vm setup pool
func (p *VMSetupPool) ReturnVMSetupByVmid(vmid string) error {
	p.Lock()
	defer p.Unlock()

	vmSetup, ok := p.OccupiedVMSetup[vmid]
	if !ok {
		return ErrNotAssign
	}
	delete(p.OccupiedVMSetup, vmid)
	p.IdleVMSetup = append(p.IdleVMSetup, vmSetup)
	return nil
}
