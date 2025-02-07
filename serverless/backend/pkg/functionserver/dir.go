// Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"). You may
// not use this file except in compliance with the License. A copy of the
// License is located at
//
//	http://aws.amazon.com/apache2.0/
//
// or in the "license" file accompanying this file. This file is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing
// permissions and limitations under the License.

package functionserver

import (
	"os"
	"path/filepath"

	"alouatta/pkg/util/identifiers"

	"github.com/pkg/errors"
)

const (
	varRunDir = "/var/run/firecracker-frakti/"
	// FirecrackerSockName is the name of the Firecracker VMM API socket
	FirecrackerSockName = "firecracker.sock"
	// FirecrackerLogFifoName is the name of the Firecracker VMM log FIFO
	FirecrackerLogFifoName = "fc-logs.fifo"
	// FirecrackerMetricsFifoName is the name of the Firecracker VMM metrics FIFO
	FirecrackerMetricsFifoName = "fc-metrics.fifo"
	// OsvOutputLogfileName is the name of the Osv log
	OsvOutputLogfileName = "osv.fifo"
)

// VMDir holds files, sockets and FIFOs scoped to a single
// VM with the given VMID. It is unique per-VM and containerd namespace.
func VMDir(namespace, vmID string) (Dir, error) {
	if err := identifiers.Validate(namespace); err != nil {
		return "", errors.Wrap(err, "invalid namespace")
	}

	if err := identifiers.Validate(vmID); err != nil {
		return "", errors.Wrap(err, "invalid vm id")
	}

	return Dir(filepath.Join(varRunDir, namespace, vmID)), nil
}

// Dir represents the root of a firecracker-frakti VM directory, which
// holds various files, sockets and FIFOs used during VM runtime.
type Dir string

// RootPath returns the top-level directory of the VM dir
func (d Dir) RootPath() string {
	return string(d)
}

// Mkdir will mkdir the RootPath with correct permissions, or no-op if it
// already exists
func (d Dir) Mkdir() error {
	return os.MkdirAll(d.RootPath(), 0700)
}

// FirecrackerSockPath returns the path to the unix socket at which the firecracker VMM
// services its API
func (d Dir) FirecrackerSockPath() string {
	return filepath.Join(d.RootPath(), FirecrackerSockName)
}

// FirecrackerLogFifoPath returns the path to the FIFO at which the firecracker VMM writes
// its logs
func (d Dir) FirecrackerLogFifoPath() string {
	return filepath.Join(d.RootPath(), FirecrackerLogFifoName)
}

// FirecrackerMetricsFifoPath returns the path to the FIFO at which the firecracker VMM writes
// metrics
func (d Dir) FirecrackerMetricsFifoPath() string {
	return filepath.Join(d.RootPath(), FirecrackerMetricsFifoName)
}

// OsvOutputLogfilePath returns the path to the log file at which the osv writes
// its logs
func (d Dir) OsvOutputLogfilePath() string {
	return filepath.Join(d.RootPath(), OsvOutputLogfileName)
}
