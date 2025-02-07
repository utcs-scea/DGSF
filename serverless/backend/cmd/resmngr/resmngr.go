package main

import (
	pb "alouatta/pb/resmngr"
	"alouatta/pkg/dataflow"
	"alouatta/pkg/dataflow/gpuschedulers"
	"alouatta/pkg/dataflow/schedulers"
	"alouatta/pkg/models"
	"alouatta/pkg/proxy"
	"alouatta/pkg/resmngr"
	"alouatta/pkg/resmngr/policy/placement"
	"alouatta/pkg/resmngr/policy/scale"
	"alouatta/pkg/util/filelog"
	"flag"
	"fmt"
	"math/rand"
	"net"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/sirupsen/logrus"

	"google.golang.org/grpc"
)

func init() {
	logFormat := os.Getenv("LOG_FORMAT")
	logLevel := os.Getenv("LOG_LEVEL")
	if strings.EqualFold(logFormat, "json") {
		logrus.SetFormatter(&logrus.JSONFormatter{
			FieldMap: logrus.FieldMap{
				logrus.FieldKeyMsg:  "message",
				logrus.FieldKeyTime: "@timestamp",
			},
			TimestampFormat: "2006-01-02T15:04:05.999Z07:00",
		})
	} else {
		logrus.SetFormatter(&logrus.TextFormatter{
			FullTimestamp: true,
		})
	}

	if level, err := logrus.ParseLevel(logLevel); err == nil {
		logrus.SetLevel(level)
	}
	if level, err := zerolog.ParseLevel(logLevel); err == nil {
		zerolog.SetGlobalLevel(level)
	} else {
		zerolog.SetGlobalLevel(zerolog.WarnLevel)
	}
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnixMicro
	zerolog.TimestampFieldName = "ts"

	filelog.SetupFileLogger()

	// pretty print
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	rand.Seed(time.Now().UnixNano())
}

//CreateSched ..
func CreateSched(name string, rescStates *resmngr.ResourceManagerStates) dataflow.Scheduler {
	log.Warn().Msgf("Using scheduler: %s", name)
	var scheduler dataflow.Scheduler
	switch name {
	case "random":
		scheduler = schedulers.NewRandomScheduler(rescStates)
	default:
		log.Error().Msgf("Scheduler %v not found", name)
		return nil
	}

	log.Warn().Msgf("returning")
	return scheduler
}

func main() {
	scalePolicyName := flag.String("scalepolicy", "maxcount", "scale policy, must match name in policies.yaml")
	placementPolicyName := flag.String("placepolicy", "loadbalance", "options: loadbalance, sameallnodes, roundrobin")
	schedulerName := flag.String("scheduler", "random", "options: random, roundrobin, forcelocality, forcemiss, best")
	flag.Parse()

	scalePolicyConfig := os.Getenv("SCALE_POL_CFG")
	if scalePolicyConfig == "" {
		scalePolicyConfig = "policies.yaml"
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", os.Getenv("RESMNGR_PORT")))
	if err != nil {
		log.Fatal().Err(err).Msg("failed to listen")
	}
	s := grpc.NewServer()

	//read policies yaml and create policy passed as parameter to us
	cfg := scale.GetConfig(scalePolicyConfig)
	var scalePol scale.Interface
	log.Warn().Msgf("Using policy (%s) with cfg (%v)", *scalePolicyName, cfg[(*scalePolicyName)].(map[interface{}]interface{}))
	switch *scalePolicyName {
	case "maxcount":
		scalePol = scale.NewMaxCountPolicy(cfg[(*scalePolicyName)].(map[interface{}]interface{}))
	case "watermark":
		scalePol = scale.NewWatermarkPolicy(cfg[(*scalePolicyName)].(map[interface{}]interface{}))
	}

	log.Warn().Msgf("Using placement policy (%s) ", *placementPolicyName)
	var placePol placement.Interface
	switch *placementPolicyName {
	case "loadbalance":
		placePol = placement.NewLoadBalance()
	case "sameallnodes":
		placePol = &placement.SameAllNodes{}
	case "roundrobin":
		placePol = placement.NewRoundRobin()
	}

	resc := models.NewResources()

	resmngrServer := &resmngr.ResourceManagerServer{
		States: &resmngr.ResourceManagerStates{
			Resources:       resc,
			ScalePolicy:     scalePol,
			PlacementPolicy: placePol,
		},
	}

	//launch resmngr
	go func() {
		log.Info().Msg("Registering grpcs")
		pb.RegisterResMngrServiceServer(s, resmngrServer)
		if err := s.Serve(lis); err != nil {
			log.Fatal().Err(err).Msg("failed to serve")
		}
	}()

	scheduler := CreateSched(*schedulerName, resmngrServer.States)
	scheduler.GetBaseScheduler().GPUScheduler = gpuschedulers.NewFirstGPUScheduler(resmngrServer.States)

	proxy.SetScheduler(scheduler)
	log.Warn().Msg("Launching backend")
	proxy.LaunchBackend(resmngrServer.States)
}
