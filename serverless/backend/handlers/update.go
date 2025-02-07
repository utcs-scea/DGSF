package handlers

import (
	"net/http"
)

// MakeUpdateHandler update specified function
func MakeUpdateHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		//we are not doing this one
		w.WriteHeader(http.StatusNotFound)
		// TODO: update is not implemented
	}
}
