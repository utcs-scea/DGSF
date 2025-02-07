// Copyright (c) OpenFaaS Author(s) 2019. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/openfaas/faas-provider/types"
	log "github.com/sirupsen/logrus"
)

// MakeReplicaUpdater updates desired count of replicas
// This would be called by an outside monitor to scale stuff up and down
// THIS IS NOT USED ATM
func MakeReplicaUpdater() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Info("update replicas")
		//we can call resmngr to spawn some replicas if we want to
	}
}

// MakeReplicaReader reads the amount of replicas for a deployment
func MakeReplicaReader() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		functionName := vars["name"]

		found := &types.FunctionStatus{}
		found.Name = functionName
		//log.Printf("Function (%s) ReplicaReader: Always tell the client theres an available replica :)", functionName)
		found.AvailableReplicas = 1

		functionBytes, _ := json.Marshal(found)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write(functionBytes)
	}
}
