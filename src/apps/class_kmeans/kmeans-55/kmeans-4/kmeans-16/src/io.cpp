//#include <io.h>
 #include "io.h"

//void read_file(struct options_t* args,
//               int*              numpoints,
//               float** input_vals) {
//
//   int dims = args->n_dims;
//   int point_number = 0;
//
//   // Open file
//   std::ifstream in;
//   in.open(args->in_file);
//   // Get num vals
//   in >> *numpoints;
//
//   // Alloc input and output arrays
//   input_vals = new float*[*numpoints];
//
//   // Read input vals
//   for (int i = 0; i < *numpoints; ++i) {
//      in >> point_number;
//      //       in >> (*input_vals)[i];
//      for (int j = 0; j < dims; j++) {
//         in >> input_vals[i][j];
//      }
//   }
//}

 void read_file(struct options_t* args,
                int*              numpoints,
                std::vector<std::vector<float>>& input_vals) {

    int dims = args->n_dims;
    int point_number = 0;

    // Open file
    std::ifstream in;
    in.open(args->in_file);
    // Get num vals
    in >> *numpoints;

    // Read input vals
    for (int i = 0; i < *numpoints; ++i) {
       in >> point_number;
       std::vector<float> row;
       float input;
       for (int j = 0; j < dims; j++) {
          in >> input;
          row.push_back(input);
       }
       input_vals.push_back(row);
    }
 }

// void read_file(struct options_t* args,
//                int*              numpoints,
//                float**             input_vals,
//                float**             output_labels,
//                float**             output_centroids) {

//    // WAT - OUTPUT VALUES PROBABLY NOT NEEDED SINCE THEY'LL BE PRINTED OUT, NOT SAVED IN A FILE

//    int dims = args->n_dims;
//    int point_number = 0;

//    // Open file
//    std::ifstream in;
//    in.open(args->in_file);
//    // Get num vals
//    in >> *numpoints;

//    // Alloc input and output arrays
// //   *input_vals = (int*) malloc(*numpoints * sizeof(int));
//    input_vals = new float*[*numpoints];
// //   *output_labels = (int*) malloc(*numpoints * sizeof(int));
//    output_labels = new float*[*numpoints];
//    for (int i = 0; i < *numpoints; ++i) {
//       input_vals[i] = new float[dims];
//    }

//    output_centroids = new float*[args->n_cluster];
//    for (int i = 0; i < args->n_cluster; ++i) {
//       output_centroids[i] = new float[dims];
// //      output_centroids[i] = new float[dims + 1];
//    }

//    // Read input vals
//    for (int i = 0; i < *numpoints; ++i) {
//       in >> point_number;
//       //       in >> (*input_vals)[i];
//       for (int j = 0; j < dims; j++) {
//          in >> input_vals[i][j];
//       }
//    }
// }
