// Copyright (c) OpenFaaS Author(s) 2019. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

package handlers

import (
	"net/http"
)

// MakeFunctionReader handler for reading functions deployed in the cluster as deployments.
func MakeFunctionReader() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		// this requires another gRPC call to resmngr to get all deployed functions
		// (NIY)
		w.WriteHeader(http.StatusOK)
		/*
			functions, err := readServices(provider, fcTool)
			if err != nil {
				log.Printf("Error getting service list: %s\n", err.Error())

				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte(err.Error()))
				return
			}

			functionBytes, _ := json.Marshal(functions)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(functionBytes)
		*/
	}
}
