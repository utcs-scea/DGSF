// Copyright (c) OpenFaaS Author(s) 2019. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

package handlers

import (
	"alouatta/pkg/resmngr"
	"context"

	"io/ioutil"
	"net/http"

	"github.com/rs/zerolog/log"
)

// MakeDeployHandler creates a handler to create new functions in the cluster

// MakeDeployHandler creates a handler to create new functions in the cluster
func MakeDeployHandler(s *resmngr.ResourceManagerStates) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Info().Msg("Handling deployment of a function")
		//this is not on the critical path, so we can get away with dialing every deploy

		defer r.Body.Close()
		body, _ := ioutil.ReadAll(r.Body)

		//i dont thinkk we need a ctx rn
		ctx := context.Background()
		reply, err := s.DeployFunction(ctx, &resmngr.DeployRequest{RequestJSON: body})

		if err != nil || reply.Ok == 0 {
			log.Fatal().Err(err).Msg("Error deploying function through grpc to ResMngr")
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
	}
}
