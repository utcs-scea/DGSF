package dataflow

import (
	"github.com/Jeffail/gabs/v2"
	"github.com/rs/zerolog/log"
)

type numericChoice struct {
	Operator     string
	Operand      int
	VariablePath string
	Next         string
}

//ChoiceState ..
type ChoiceState struct {
	Choices     []numericChoice //TODO: this should be a base class Choice or something
	outChannels map[string]chan<- InterStateMessage
	BaseState
}

//NewChoiceState ..
func NewChoiceState(name string, json *gabs.Container) *ChoiceState {
	var choicesList []numericChoice
	children := json.ChildrenMap()
	if _, ok := children["Choices"]; !ok {
		log.Error().Msg("Choice state doesn't have Choices required field")
	}

	//log.Debug().Msgf("Choices: %v", json.StringIndent("", "  "))

	//theres A LOT of cases here, we only do numeric for now
	//and no default (which is outside of the Choices array)
	for _, choice := range json.S("Choices").Children() {
		//log.Debug().Msgf("Choice: %v", choice.StringIndent("", "  "))
		//check what condition this choice has
		if choice.Exists("NumericEquals") {
			cond := choice.S("NumericEquals").Data().(float64)
			//TODO: error checking if path exists
			next := choice.S("Next").Data().(string)
			variable := choice.S("Variable").Data().(string)
			nc := numericChoice{Operator: "equal", Operand: int(cond), Next: next, VariablePath: variable}
			choicesList = append(choicesList, nc)
			//log.Debug().Msgf("found numeric equals choice:  = %v", variable)
		}
		//TODO: other conditions that we dont use
	}

	return &ChoiceState{
		Choices:     choicesList,
		outChannels: make(map[string]chan<- InterStateMessage),
		BaseState: BaseState{
			Name:      name,
			NextState: "",
			IsEnd:     false,
		},
	}
}

//GetPossibleNextStates returns all possible states we can go to
func (p *ChoiceState) GetPossibleNextStates() []string {
	s := make(map[string]struct{})
	var exists = struct{}{}

	//remove duplicates by using a hacked set
	for _, choice := range p.Choices {
		s[choice.Next] = exists
	}

	//transform into list
	var ret []string
	for k := range s {
		//log.Debug().Msgf("building possible choice states list, adding %v", k)
		ret = append(ret, k)
	}
	return ret
}

//SetNextStateChannel set the channel of a next state. caller is responsible for setting this
//for all possible states
func (p *ChoiceState) SetNextStateChannel(state string, c chan<- InterStateMessage) {
	//log.Debug().Msgf("setting channel of choice state %v", state)
	p.outChannels[state] = c
}

//Execute for ChoiceState
//TODO: we can make this loop in the base class State
func (p *ChoiceState) Execute() {
	for {
		select {
		case inputMsg := <-p.In:
			p.executeMessage(inputMsg)
		case <-p.done:
			return
		}
	}
}

func (p *ChoiceState) executeMessage(inputMsg InterStateMessage) {
	json, err := gabs.ParseJSON([]byte(inputMsg.Body))
	if err != nil {
		log.Error().
			Str("json", inputMsg.Body).
			Err(err).
			Msg("error parsing json choice input")
	}

	//log.Debug().Msgf("executing choice state, input: %v", json.StringIndent("", "  "))

	for _, choice := range p.Choices {
		//log.Debug().Msgf("tryng choice %v %v", choice.Operator, choice.Operand)

		//NumericEquals
		if choice.Operator == "equal" {
			path := choice.VariablePath[2:] //this removes "$."
			//log.Debug().Msgf("operand path (%v)", path)
			if value, ok := json.Path(path).Data().(float64); ok {
				//log.Debug().Msgf("comparing jsons %v with %v", int(value), choice.Operand)
				if int(value) == choice.Operand {
					//this is it, send message and get out
					//log.Debug().Msgf("Choice matches, sending message through channel to (%v)", choice.Next)
					p.outChannels[choice.Next] <- inputMsg
					return
				}
			} else {
				log.Error().Msgf("Cant find path (%v) on input json %v", path, json.StringIndent("", "  "))
			}
		}
		//other operators can go here, preferably with a base class and a compare method
	}
}

//GetType ...
func (p *ChoiceState) GetType() string {
	return "Choice"
}

//Clone creates a copy
func (p *ChoiceState) Clone() State {
	c := *p
	return &c
}
