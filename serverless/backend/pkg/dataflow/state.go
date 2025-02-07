package dataflow

//State ..
type State interface {
	Execute()
	GetBaseState() *BaseState
	Clone() State
	GetType() string
}

//BaseState ..
type BaseState struct {
	Name          string
	NextState     string //could be a pointer later
	DataFlowId    string // for tracing
	NumFanOut     int    // for rebalance on task state; assigned to df under map state
	NextStateIntr State
	IsEnd         bool
	In            chan InterStateMessage   //receive only channel
	Out           chan<- InterStateMessage //send only channel
	done          <-chan struct{}          //channel to close when done
	State
}

//GetBaseState ..
func (b *BaseState) GetBaseState() *BaseState {
	return b
}

//SetDoneChannel ..
func (b *BaseState) SetDoneChannel(done <-chan struct{}) {
	b.done = done
}

//GetInputChannel get the input channel, create if not exists
func (b *BaseState) GetInputChannel() chan InterStateMessage {
	if b.In == nil {
		b.In = make(chan InterStateMessage)
	}
	return b.In
}

//SetOutputChannel ..
func (b *BaseState) SetOutputChannel(out chan<- InterStateMessage) {
	b.Out = out
}

//ConnectTo connect two states by a channel
func (b *BaseState) ConnectTo(dest *BaseState) {
	c := make(chan InterStateMessage)
	b.Out = c
	dest.In = c
}

//ConnectInitial creates a channel, sets as in for a state, and return send only chan
func (b *BaseState) ConnectInitial() chan<- InterStateMessage {
	c := make(chan InterStateMessage)
	b.In = c
	return c
}

//ConnectLast creates channel, sets as out for state, returns recv only
func (b *BaseState) ConnectLast() <-chan InterStateMessage {
	c := make(chan InterStateMessage)
	b.Out = c
	return c
}
