package dataflow

//Dataflow ..
type Dataflow struct {
	StartState string
	States     map[string]State
}

func (d *Dataflow) createState(name, dfid string, numFanOut int, done chan struct{}) State {
	state := d.States[name].Clone()
	state.GetBaseState().DataFlowId = dfid
	state.GetBaseState().NumFanOut = numFanOut
	state.GetBaseState().SetDoneChannel(done)
	return state
}

//Execute create all states and connect them
//current limitations (that might actually not be true):
// * only one final state
// * first state cannot be Choice
// * branches/choices are really sketchy to create with the original design
func (d *Dataflow) Execute(inputData string, schedulerChan chan<- SchedulableMessage,
	dfid string, numFanOut int) string {
	//create end notification channel
	doneChan := make(chan struct{})

	createdStates := make(map[string]State)
	//clone the dataflow (all states)
	for name := range d.States {
		createdStates[name] = d.createState(name, dfid, numFanOut, doneChan)
	}

	//connect to first state
	inputChan := createdStates[d.StartState].GetBaseState().GetInputChannel()
	//create out channel
	outputChan := make(chan InterStateMessage)

	//connect all of the states (not in order nor following the graph)
	for _, state := range createdStates {
		//if state is final connect last
		if state.GetBaseState().IsEnd {
			state.GetBaseState().SetOutputChannel(outputChan)
		}

		if state.GetType() == "Choice" {
			cstate := state.(*ChoiceState)
			nexts := cstate.GetPossibleNextStates()
			for _, next := range nexts {
				nextState := createdStates[next]
				c := nextState.GetBaseState().GetInputChannel()
				//inform Choice of next state's input channel
				cstate.SetNextStateChannel(next, c)
			}
		} else {
			//if we dont have a next, we dont need to set this up
			next := state.GetBaseState().NextState
			if next != "" {
				nextState := createdStates[next]
				//connect the two states
				c := nextState.GetBaseState().GetInputChannel()
				state.GetBaseState().SetOutputChannel(c)
				state.GetBaseState().NextStateIntr = nextState
			}
		}
	}

	//start 'em all
	for _, state := range createdStates {
		go state.Execute()
	}

	//log.Debug().Msg("finished creating dataflow, running")

	inputMessage := InterStateMessage{
		SchedulerChan: schedulerChan,
		Body:          inputData,
	}

	//start execution and wait until finish
	inputChan <- inputMessage
	outputMsg := <-outputChan

	//log.Debug().Msg("dataflow execution is completed, closing channel")
	close(doneChan)

	//log.Debug().Str("Body resp from last fn of df", outputMsg.Body).Msg("")
	return outputMsg.Body
}
