package gpu

import (
	"alouatta/pkg/models"
)

//GPUSelection is the GPU scheduling policy interface
type GPUSelection interface {
	OnChooseGPU(gpuMemReq uint32, gpuNodes []*models.GPUNode, numWorkerPerGPU uint32) (string, uint32, *models.GPUNode)
}
