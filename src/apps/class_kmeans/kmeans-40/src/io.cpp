#include <io.h>

void read_file(struct options_t* args,
               REAL**            input_dataset,
               int**             output_labels,
               REAL**            output_centroids) {

  	// Open file
	std::ifstream in;
	in.open(args->in_file);
	// Get num vals
	in >> args->num_points;
	//printf("n_vals: %d", args->num_points);

	// Alloc input and output arrays
	*input_dataset = (REAL*) malloc(args->num_points * sizeof(REAL) * args->dims);
	*output_labels = (int*) malloc(args->num_points * sizeof(int));
	*output_centroids = (REAL*) malloc(args->num_cluster * sizeof(REAL) * args->dims);

	// Read input vals
	int cnt;
	for (int i = 0; i < args->num_points; ++i) {
		in >> cnt;
		//printf("cnt %d: %d", i, cnt);
		for (int d = 0; d < args->dims; ++d) {
			in >> (*input_dataset)[i*args->dims+d];
			//printf("data %d: float %f, double %lf\n", d, float((*input_dataset)[i*args->dims+d]), double((*input_dataset)[i*args->dims+d]));

		}
		//printf("\n");
	}
}

