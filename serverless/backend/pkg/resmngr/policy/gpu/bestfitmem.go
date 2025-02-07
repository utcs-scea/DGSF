package gpu

import (
	"alouatta/pkg/models"
)

//BestFitMemory struct for bestfit memory policy
type BestFitMemory struct{}

//OnChooseGPU implements best fit policy regarding memory
func (s *BestFitMemory) OnChooseGPU(gpuMemReq uint32, gpuNodes []*models.GPUNode, numWorkerPerGPU uint32) (string, uint32, int, *models.GPUNode) {

	return "", 0, 0, nil

	/*
		minMemLeft := uint32(4294967295) // init to max of uint32
		bestUUID := ""
		var bestGPUNode *models.GPUNode
		var workerPort uint32
		var bestGPUInfo *models.GPUInfo
		workerPort = 0
		gpuIdx := 0
		//do a quick best fit for the GPUs attached to this node
		for _, gpuNode := range gpuNodes {
			gpuNode.Lock()
			for _, gpuInfo := range gpuNode.GPUInfos {
				uuid := gpuInfo.UUID
				totalMemory := gpuInfo.Memory
				_, exists := gpuNode.AllocatedGPUResc[uuid]
				if !exists {
					worker := make([]bool, numWorkerPerGPU)
					for i := range worker {
						worker[i] = false
					}
					gpuNode.AllocatedGPUResc[uuid] = &models.GPUResc{
						AllocatedMem:    0,
						AllocatedWorker: worker,
					}
				}
				memLeft := totalMemory - gpuNode.AllocatedGPUResc[uuid].AllocatedMem
				//if it fits and is better than what we had, update
				if memLeft >= gpuMemReq && minMemLeft > memLeft {
					minMemLeft = memLeft
					bestUUID = uuid
					bestGPUNode = gpuNode
					bestGPUInfo = gpuInfo
				}
			}
			gpuNode.Unlock()
		}
		if bestUUID != "" {
			bestGPUNode.Lock()
			defer bestGPUNode.Unlock()
			resc := bestGPUNode.AllocatedGPUResc[bestUUID]
			resc.AllocatedMem += gpuMemReq
			found := false

			for i := range resc.AllocatedWorker {
				if !resc.AllocatedWorker[i] {
					workerPort = uint32(i) + bestGPUInfo.FirstWorkerPort
					resc.AllocatedWorker[i] = true
					gpuIdx = i
					found = true
					break
				}
			}
			if !found {
				log.Fatal().Msg("Found GPU with available memory but not worker")
			}
		}
		return bestUUID, workerPort, gpuIdx, bestGPUNode
	*/
}
