package main

import (
	pb "alouatta/pb/functionserver"
	resmngrpb "alouatta/pb/resmngr"
	fnserver "alouatta/pkg/functionserver"
	llog "alouatta/pkg/util/log"
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gofrs/uuid"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

type functionServer struct {
	pb.UnimplementedFunctionServer
	sync.RWMutex
	fcTool         *fnserver.Tool
	newMachineLock sync.Mutex
	ctx            context.Context
	vmSetupPool    *fnserver.VMSetupPool
	state          fnserver.ServiceState

	fnCountLock sync.Mutex
	fnCount     map[string]uint32
}

var vmCount uint32

func (s *functionServer) CreateReplica(ctx context.Context, req *pb.CreateRequest) (*pb.CreateReply, error) {
	log.Debug().Str("fnName", req.ImageName).Msg("Launching VM...")

	name := filepath.Base(req.ImageName)
	name = strings.TrimSuffix(name, filepath.Ext(name))

	//book keep count of fns spawned
	s.fnCountLock.Lock()
	_, exists := s.fnCount[name]
	if !exists {
		s.fnCount[name] = 0
	}
	s.fnCount[name]++
	s.fnCountLock.Unlock()

	replica, err := fnserver.CreateReplica(s.ctx, &s.newMachineLock,
		s.fcTool.FirecrackerBinary, s.vmSetupPool, &s.state, req)
	if err != nil {
		log.Fatal().Err(err).Msg("err on CreateReplica")
	}

	atomic.AddUint32(&vmCount, 1)
	vmid := replica.Vmid
	ip := replica.MachineConfig.
		NetworkInterfaces[0].
		StaticConfiguration.
		IPConfiguration.IPAddr.IP.String()

	log.Warn().
		Str("ip", ip).
		Str("vmid", vmid).
		Uint32("vmCount", atomic.LoadUint32(&vmCount)).
		Str("fnName", name).
		Msg("VM launched and ready")

	if err == nil {
		return &pb.CreateReply{
			Ip:   ip,
			Vmid: vmid,
		}, nil
	}
	return nil, err
}

func (s *functionServer) ReleaseVMSetup(ctx context.Context, req *pb.ReleaseRequest) (*empty.Empty, error) {
	vmid := req.Vmid
	err := s.vmSetupPool.ReturnVMSetupByVmid(vmid)
	if err != nil {
		return nil, err
	}
	return &empty.Empty{}, nil
}

// getOutboundIP get preferred outbound ip of this machine
func getOutboundIP() net.IP {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		log.Fatal().Err(err).Msg("dial googld dns")
	}
	defer conn.Close()
	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP
}

func init() {
	logLevel := os.Getenv("LOG_LEVEL")
	if level, err := zerolog.ParseLevel(logLevel); err == nil {
		zerolog.SetGlobalLevel(level)
	} else {
		zerolog.SetGlobalLevel(zerolog.WarnLevel)
	}
	//pretty print
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	if llevel, err := llog.ParseLevel(logLevel); err == nil {
		logrus.SetLevel(llevel)
	} else {
		logrus.SetLevel(logrus.WarnLevel)
	}
}

func startProxy() {
	//references:
	//https://golang.org/pkg/net/http/#RoundTripper
	//https://www.integralist.co.uk/posts/golang-reverse-proxy/
	director := func(req *http.Request) {
		//log.Printf("headers  %v", req.Header)
		//req.Header.Add("X-Forwarded-Host", req.Host)
		req.URL.Scheme = "http"
		req.URL.Host = req.Header.Get("Redirect-To") + ":8080"
		log.Info().
			Str("from", req.Host).
			Str("Redirect-To", req.URL.Host).
			Msgf("reverse proxy redirecting request to function")
	}

	dialer := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   1500 * time.Second,
			KeepAlive: -1,
		}).DialContext,
		ForceAttemptHTTP2:     false,
		MaxIdleConns:          100,
		IdleConnTimeout:       0,
		TLSHandshakeTimeout:   0,
		ExpectContinueTimeout: 0,
	}

	proxy := &httputil.ReverseProxy{Director: director, Transport: dialer}
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		proxy.ServeHTTP(w, r)
	})
	go http.ListenAndServe(fmt.Sprintf(":%s", os.Getenv("FUNCTIONPROXY_PORT")), nil)
}

func main() {
	vmCount = 0
	netname := flag.String("netname", "fcnet", "network name of cni")
	sizeIPPool := flag.Int("ippool", 0, "how many ips to pre allocate")
	cachePort := flag.Uint("cacheport", 6379, "port of the cache each vm will talk to")
	avaLog := flag.String("avaLog", "info", "ava log level")
	flag.Parse()

	listenPort := os.Getenv("FUNCTIONSERVER_PORT")
	resmngrIP := os.Getenv("RESMNGR_ADDR")
	resmngrPort := os.Getenv("RESMNGR_PORT")

	log.Warn().Msgf("listenPort %v", listenPort)
	log.Debug().Msgf("resmngrIP %v", resmngrIP)

	//start reverse proxy
	startProxy()

	fcTool, err := fnserver.NewTool()
	if err != nil {
		log.Fatal().Err(err).Msg("fail to create firecracker tool")
	}

	server := &functionServer{
		fcTool: fcTool,
		ctx:    context.Background(),
		state: fnserver.ServiceState{
			NetName:     *netname,
			CachePort:   *cachePort,
			AvALogLevel: *avaLog,
		},
		fnCount: make(map[string]uint32),
	}

	vmSetupPool := fnserver.NewVMSetupPool(server.ctx, server.state.NetName)
	err = vmSetupPool.FillVMSetupPool(*sizeIPPool)
	if err != nil {
		log.Fatal().Err(err).Msg("fail to create vm setup pool")
	}
	server.vmSetupPool = vmSetupPool

	// get current ip and register node
	outboundIP := getOutboundIP().String()
	server.state.ResmngrAddr = fmt.Sprintf("%s:%s", resmngrIP, resmngrPort)
	server.state.FcdaemonAddr = fmt.Sprintf("%s:%s", outboundIP, listenPort)

	go func() {
		log.Warn().Msgf("Connecting to resmngr at %v", server.state.ResmngrAddr)
		log.Warn().Msgf("Bound to IP %s", server.state.FcdaemonAddr)

		var opts []grpc.DialOption
		opts = append(opts, grpc.WithInsecure())
		opts = append(opts, grpc.WithBlock())

		conn, err := grpc.Dial(server.state.ResmngrAddr, opts...)
		defer conn.Close()
		if err != nil {
			log.Fatal().Err(err).Msg("gRPC dial to resmngr failed")
		}

		//create grpc client
		client := resmngrpb.NewResMngrServiceClient(conn)

		//register ourself with resmngr
		uuid, _ := uuid.NewV4()
		listenPortInt, _ := strconv.Atoi(listenPort)
		req := &resmngrpb.RegisterFunctionNodeRequest{
			Ip:     outboundIP,
			Port:   uint32(listenPortInt),
			NodeId: uuid.String(),
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, err = client.RegisterFunctionNode(ctx, req)
		if err != nil {
			log.Fatal().Err(err).Msg("fail to register node")
		}
		log.Warn().Msg("Registered at resmngr")
	}()

	// setup listener
	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", listenPort))
	if err != nil {
		log.Fatal().Err(err).Msg("failed to listen")
	}

	log.Warn().Msgf("Launching our gRPC server on port %v", listenPort)
	grpcServer := grpc.NewServer()
	pb.RegisterFunctionServer(grpcServer, server)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatal().Err(err).Msg("failed to serve")
	}
}
