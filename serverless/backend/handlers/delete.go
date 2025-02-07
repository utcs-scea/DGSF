// Copyright (c) OpenFaaS Author(s) 2019. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

package handlers

import (
	"net/http"
)

// MakeDeleteHandler delete a function
func MakeDeleteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		//We're not gonna use this, but this should call a gRPC to resmngr
		w.WriteHeader(http.StatusBadRequest)
		/*
			log.Info("delete request")
			defer r.Body.Close()

			body, _ := ioutil.ReadAll(r.Body)
			request := requests.DeleteFunctionRequest{}
			if err := json.Unmarshal(body, &request); err != nil {
				log.Errorf("error de-serializing request body:%s", body)
				log.Error(err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}

			if len(request.FunctionName) == 0 {
				log.Errorln("can not delete a function, request function name is empty")
				w.WriteHeader(http.StatusBadRequest)
				return
			}

			provider.FLock.Lock()
			defer provider.FLock.Unlock()
			delete(provider.Functions, request.FunctionName)

			log.Infof("delete request %s successful", request.FunctionName)
		*/
	}
}
