// Package proxy provides a default function invocation proxy method for OpenFaaS providers.
//
// The function proxy logic is used by the Gateway when `direct_functions` is set to false.
// This means that the provider will direct call the function and return the results.  This
// involves resolving the function by name and then copying the result into the original HTTP
// request.
//
// openfaas-provider has implemented a standard HTTP HandlerFunc that will handle setting
// timeout values, parsing the request path, and copying the request/response correctly.
// 		bootstrapHandlers := bootTypes.FaaSHandlers{
// 			FunctionProxy:  proxy.NewHandlerFunc(timeout, resolver),
// 			DeleteHandler:  handlers.MakeDeleteHandler(clientset),
// 			DeployHandler:  handlers.MakeDeployHandler(clientset),
// 			FunctionReader: handlers.MakeFunctionReader(clientset),
// 			ReplicaReader:  handlers.MakeReplicaReader(clientset),
// 			ReplicaUpdater: handlers.MakeReplicaUpdater(clientset),
// 			InfoHandler:    handlers.MakeInfoHandler(),
// 		}
//
// proxy.NewHandlerFunc is optional, but does simplify the logic of your provider.
package proxy

import (
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/rs/zerolog/log"

	"alouatta/handlers"
	"alouatta/pkg/dataflow"
	"alouatta/pkg/resmngr"

	"github.com/gorilla/mux"
	"github.com/openfaas/faas-provider/httputil"

	ftypes "alouatta/types"

	bootstrap "github.com/openfaas/faas-provider"
	bootTypes "github.com/openfaas/faas-provider/types"
)

const (
	defaultContentType     = "text/plain"
	errMissingFunctionName = "Please provide a valid route /function/function_name."
)

// BaseURLResolver URL resolver for proxy requests
//
// The FaaS provider implementation is responsible for providing the resolver function implementation.
// BaseURLResolver.Resolve will receive the function name and should return the URL of the
// function service.

var scheduler dataflow.Scheduler

//SetScheduler ..
func SetScheduler(s dataflow.Scheduler) {
	scheduler = s
}

// NewHandlerFunc creates a standard http.HandlerFunc to proxy function requests.
// The returned http.HandlerFunc will ensure:
//
// 	- proper proxy request timeouts
// 	- proxy requests for GET, POST, PATCH, PUT, and DELETE
// 	- path parsing including support for extracing the function name, sub-paths, and query paremeters
// 	- passing and setting the `X-Forwarded-Host` and `X-Forwarded-For` headers
// 	- logging errors and proxy request timing to stdout
//
// Note that this will panic if `resolver` is nil.
func NewHandlerFunc(config bootTypes.FaaSConfig) http.HandlerFunc {
	if scheduler == nil {
		panic("NewHandlerFunc: empty proxy handler scheduler, cannot be nil")
	}

	proxyClient := NewProxyClientFromConfig(config)
	//we have to give the scheduler the http client to use when sending requests
	scheduler.SetProxyClient(proxyClient)

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			defer r.Body.Close()
		}

		switch r.Method {
		case http.MethodPost:
			//this is hacky, but works for us. check if POST url path starts with dataflow
			if strings.HasPrefix(r.URL.Path, "/function/dataflow") {
				scheduleDataflow(w, r, proxyClient, scheduler)
			} else if strings.HasPrefix(r.URL.Path, "/function/regdf") {
				registerDataflow(w, r, proxyClient, scheduler)
			} else {
				scheduleSingleRequest(w, r, proxyClient, scheduler)
			}

		case http.MethodPut,
			http.MethodPatch,
			http.MethodDelete,
			http.MethodGet:

			scheduleSingleRequest(w, r, proxyClient, scheduler)

		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}
}

// NewProxyClientFromConfig creates a new http.Client designed for proxying requests and enforcing
// certain minimum configuration values.
func NewProxyClientFromConfig(config bootTypes.FaaSConfig) *http.Client {
	return NewProxyClient(config.GetReadTimeout(), config.GetMaxIdleConns(), config.GetMaxIdleConnsPerHost())
}

// NewProxyClient creates a new http.Client designed for proxying requests, this is exposed as a
// convenience method for internal or advanced uses. Most people should use NewProxyClientFromConfig.
func NewProxyClient(timeout time.Duration, maxIdleConns int, maxIdleConnsPerHost int) *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:       timeout,
				KeepAlive:     -1,
				FallbackDelay: -1,
				DualStack:     true,
			}).DialContext,
			MaxIdleConns:          maxIdleConns,
			MaxIdleConnsPerHost:   maxIdleConnsPerHost,
			IdleConnTimeout:       0,
			TLSHandshakeTimeout:   0,
			ExpectContinueTimeout: 1500 * time.Millisecond,
		},
		Timeout: timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

// scheduleRequest schedules the function(s) and proxy them
func scheduleSingleRequest(w http.ResponseWriter, originalReq *http.Request,
	proxyClient *http.Client, scheduler dataflow.Scheduler) {

	pathVars := mux.Vars(originalReq)
	//the "name" arg is defined all the way in serve.go
	//the ":" prefix is to pretend we have an ARN
	functionName := ":" + pathVars["name"]
	//log.Debug().Str("fn", functionName).
	//	Str("url", originalReq.URL.String()).
	//	Msg("in proxyRequest")

	if functionName == "" {
		httputil.Errorf(w, http.StatusBadRequest, errMissingFunctionName)
		return
	}

	//read input from user's request
	defer originalReq.Body.Close()
	body, _ := ioutil.ReadAll(originalReq.Body)

	out, err := scheduler.RunSingleFunction(functionName, string(body), nil, "", false)
	if err != nil {
		log.Error().Msg("error while scheduling single function")
		httputil.Errorf(w, http.StatusServiceUnavailable, "Error running single function %v\n", err)
	}

	//send response
	SendResponse(w, strings.NewReader(out.Body))
}

func registerDataflow(w http.ResponseWriter, r *http.Request,
	proxyClient *http.Client, scheduler dataflow.Scheduler) {

	dfName := strings.TrimPrefix(r.URL.Path, "/function/regdf/")

	//https://www.alexedwards.net/blog/how-to-properly-parse-a-json-request-body
	//decode the json body; limit json size
	r.Body = http.MaxBytesReader(w, r.Body, 1048576)
	buf, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Error().Err(err).Msg("error reading request body")
		httputil.Errorf(w, http.StatusServiceUnavailable, "Error reading request body\n")
	}

	log.Printf("registering dataflow (%v)", dfName)
	err = scheduler.RegisterDataFlow(dfName, buf)

	if err != nil {
		httputil.Errorf(w, http.StatusServiceUnavailable, "Error parsing json: %v\n", err)
	}
}

func scheduleDataflow(w http.ResponseWriter, r *http.Request,
	proxyClient *http.Client, scheduler dataflow.Scheduler) {
	//TODO: maybe we will need input/output parsing, but we can just pass the entire data json around
	//https://docs.aws.amazon.com/step-functions/latest/dg/input-output-example.html
	dfName := strings.TrimPrefix(r.URL.Path, "/function/dataflow/")

	r.Body = http.MaxBytesReader(w, r.Body, 1048576)
	buf, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Error().Err(err).Msg("error reading request body")
		httputil.Errorf(w, http.StatusServiceUnavailable, "Error reading request body\n")
	}
	initialData := string(buf)

	output, err := scheduler.RunDataflow(dfName, initialData)
	if err != nil {
		httputil.Errorf(w, http.StatusInternalServerError, "Dataflow is not registered\n")
	}

	SendResponse(w, strings.NewReader(output))
}

//SendResponse ..
func SendResponse(w http.ResponseWriter, bodyReader io.Reader) error {
	//we might have to change this later, right now EVERYTHING is json
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	//because we read the body in the task, we have to do this thing, sucks..
	io.Copy(w, bodyReader)

	return nil
}

//LaunchBackend launches web server to handle requests
func LaunchBackend(s *resmngr.ResourceManagerStates) {
	readConfig := ftypes.ReadConfig{}
	osEnv := ftypes.OsEnv{}
	cfg := readConfig.Read(osEnv)

	bootstrapConfig := bootTypes.FaaSConfig{
		ReadTimeout:     cfg.ReadTimeout,
		WriteTimeout:    cfg.WriteTimeout,
		TCPPort:         &cfg.Port,
		EnableHealth:    true,
		EnableBasicAuth: false,
	}
	log.Info().Dur("HTTP Read Timeout", bootstrapConfig.GetReadTimeout()).Msg("")
	log.Info().Dur("HTTP Write Timeout", bootstrapConfig.WriteTimeout).Msg("")

	bootstrapHandlers := bootTypes.FaaSHandlers{
		FunctionProxy:  NewHandlerFunc(bootstrapConfig),
		DeleteHandler:  handlers.MakeDeleteHandler(),
		DeployHandler:  handlers.MakeDeployHandler(s),
		FunctionReader: handlers.MakeFunctionReader(),
		ReplicaReader:  handlers.MakeReplicaReader(),
		ReplicaUpdater: handlers.MakeReplicaUpdater(),
		UpdateHandler:  handlers.MakeUpdateHandler(),
		HealthHandler:  handlers.MakeHealthHandler(),
	}

	log.Info().Int("port", cfg.Port).Msg("Listening on")
	bootstrap.Serve(&bootstrapHandlers, &bootstrapConfig)
}
