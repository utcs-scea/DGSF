package dataflow

//SchedulerHint ..
type SchedulerHint struct {
	Reads, Uploads  int
	ScheduledNodeIP string
	FunctionID      string
}
