package scale

import (
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v2"
	"io/ioutil"
)

//Interface is the scaling policy interface
type Interface interface {
	//OnFunctionDeploy scale by some amount on fn deploy
	OnFunctionDeploy(int, int) uint32
	//OnRequestNoneAvailable returns >0 if should wait or 0 if error
	OnRequestNoneAvailable() uint32
	//OnFunctionCompletion returns if VM should die or not
	OnFunctionCompletion() (bool, uint32)
}

//GetConfig returns the policy configuration from input file
func GetConfig(fname string) map[interface{}]interface{} {
	data, err := ioutil.ReadFile(fname)
	if err != nil {
		log.Fatal().Err(err).Msgf("error parsing yaml %v", fname)
	}

	m := make(map[interface{}]interface{})
	err = yaml.Unmarshal([]byte(data), &m)
	if err != nil {
		log.Fatal().Err(err).Msg("err in yaml unmarshal")
	}

	//log.Printf("--- m:\n%v\n\n", m)
	return m
}
