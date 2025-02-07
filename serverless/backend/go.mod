module alouatta

go 1.13

require (
	github.com/Jeffail/gabs/v2 v2.5.0
	github.com/firecracker-microvm/firecracker-go-sdk v0.22.0
	github.com/gofrs/uuid v3.3.0+incompatible
	github.com/golang/protobuf v1.5.2
	github.com/gorilla/mux v1.7.1
	github.com/openfaas/faas-provider v0.15.1
	github.com/pkg/errors v0.9.1
	github.com/rs/zerolog v1.19.0
	github.com/sirupsen/logrus v1.8.0
	golang.org/x/net v0.0.0-20210405180319-a5a99cb37ef4 // indirect
	golang.org/x/sync v0.0.0-20210220032951-036812b2e83c
	golang.org/x/sys v0.0.0-20210510120138-977fb7262007 // indirect
	golang.org/x/tools v0.1.5 // indirect
	golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1 // indirect
	google.golang.org/grpc v1.38.0
	google.golang.org/protobuf v1.26.0
	gopkg.in/yaml.v2 v2.4.0
	gotest.tools v2.2.0+incompatible
)

replace github.com/firecracker-microvm/firecracker-go-sdk => ./third_party/firecracker-go-sdk
