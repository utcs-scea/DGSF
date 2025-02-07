package dataflow

//InterStateMessage ..
type InterStateMessage struct {
	Body          string
	SchedulerChan chan<- SchedulableMessage
	Hints         SchedulerHint
}
