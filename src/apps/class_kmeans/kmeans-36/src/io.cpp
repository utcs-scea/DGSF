#include "io.h"
#include "common.h"

void read_file(struct options_t* args, int* n_points, real** points) {

  // Open file
  std::ifstream in;
  in.open(args->in_file);
  // Get num vals
  in >> *n_points;

  // Alloc input and output arrays
  *points = (real*) malloc(*n_points * args->dimensions * sizeof(real));

  int index;

  // Read input vals
  for (int i = 0; i < (*n_points) * args->dimensions; ++i) {
    in >> index;
    in >> (*points)[i];
  }

  DEBUG_OUT(index);
  DEBUG_OUT((*points)[0]);
  DEBUG_OUT((*points)[*n_points * args->dimensions - 1]);
}
