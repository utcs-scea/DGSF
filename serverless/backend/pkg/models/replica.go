package models

// VMStatus is an int represents vm status
type VMStatus int

const (
	VMIdle          VMStatus = 0
	VMAllocated     VMStatus = 1
	VMCreating      VMStatus = 2
	VMDead          VMStatus = 13
	VMStarting      VMStatus = 200
	VMStartingLimit VMStatus = 210
)

// Replica contains the metadata state of serverless worker
type Replica struct {
	Vmid              string
	IP                string
	Function          string
	IdleExecutionEnvs int
	Status            VMStatus
	DeployNode        *Node
}
