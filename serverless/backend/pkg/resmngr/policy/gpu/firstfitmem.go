package gpu

import (
	"alouatta/pkg/models"
)

//FirstFitMemory struct for first fit memory policy
type FirstFitMemory struct{}

//OnChooseGPU implements first fit policy regarding memory
func (s *FirstFitMemory) OnChooseGPU(gpuMemReq uint32, gpuNodes []*models.GPUNode, numWorkerPerGPU uint32) (string, uint32, int, *models.GPUNode) {
	//TODO

	return "", 0, 0, nil
	/*
		foundUUID := ""
		var foundGPUNode *models.GPUNode
		var workerPort uint32
		workerPort = 0
		gpuIdx := 0
		for _, gpuNode := range gpuNodes {
			gpuNode.Lock()
			for _, gpuInfo := range gpuNode.GPUInfos {
				//if fits, return
				totalMemory := gpuInfo.Memory
				uuid := gpuInfo.UUID
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
				resc := gpuNode.AllocatedGPUResc[uuid]
				if totalMemory-resc.AllocatedMem >= gpuMemReq {
					foundUUID = uuid
					foundGPUNode = gpuNode
					resc.AllocatedMem += gpuMemReq
					found := false
					for i := range resc.AllocatedWorker {
						if !resc.AllocatedWorker[i] {
							workerPort = uint32(i) + gpuInfo.FirstWorkerPort
							resc.AllocatedWorker[i] = true
							found = true
							gpuIdx = i
							break
						}
					}
					gpuNode.Unlock()
					if !found {
						log.Fatal().Msg("Found GPU with available memory but not worker")
					}
					return foundUUID, workerPort, gpuIdx, foundGPUNode
				}
			}
			gpuNode.Unlock()
		}
		return "", workerPort, gpuIdx, nil
	*/
}
