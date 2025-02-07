all: lint gen-proto build-functionserver build-resmngr fc-go-sdk

.PHONY: fc-go-sdk
fc-go-sdk: 
	make -C third_party/firecracker-go-sdk

.PHONY: build-functionserver
build-functionserver:
	go build -o functionserver cmd/functionserver/functionserver.go

.PHONY: build-resmngr
build-resmngr:
	go build -o resmngr cmd/resmngr/resmngr.go

.PHONY: gen-proto
gen-proto:
	protoc pb/functionserver/functionserver.proto --go-grpc_out=pb/functionserver --go_out=pb/functionserver
	protoc pb/resmngr/resmngr.proto --go-grpc_out=pb/resmngr --go_out=pb/resmngr

.PHONY: clean
clean:
	rm -rf functionserver || true
	rm -rf resmngr || true

.PHONY: vet
vet:
	@echo "go vet ."
	@go vet $$(go list ./... | grep -v third_party/) ; if [ $$? -eq 1 ]; then \
		echo ""; \
		echo "Vet found suspicious constructs. Please check the reported constructs"; \
		echo "and fix them if necessary before submitting the code for review."; \
		exit 1; \
	fi

.PHONY: lint
lint:
	@echo "golint ."
	@go get golang.org/x/tools/cmd/goimports
	@golint -set_exit_status handlers/ pkg/ types/ ; if [ $$? -eq 1 ]; then \
		echo ""; \
		echo "Lint found errors in the source code. Please check the reported errors"; \
		echo "and fix them if necessary before submitting the code for review."; \
		exit 1; \
	fi
