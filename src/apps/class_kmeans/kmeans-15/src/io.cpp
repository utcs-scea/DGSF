#include <io.h>
#include <string>
#include <sstream>

void read_file(struct options_t* args,
               int*              n_vals,
               double***         input_vals,
               int*              n_dims)
{

  	// Open file
	std::ifstream in;
	in.open(args->in_file);
	// Get num vals
	in >> *n_vals;
    //std::cout<<"sud "<<*n_vals<<*n_dims<<std::endl;
	// Alloc input and output arrays
    //std::cout<<"sud "<<line<<std::endl;
	*input_vals = (double**) malloc((*n_vals) * sizeof(double));
	// Read input vals
	for (int i = 0; i < *n_vals; ++i) {
        //getline(in,line);
        int a;
        in >> a;
        (*input_vals)[i] = (double*) malloc((*n_dims) * sizeof(double));
        for (int j=0; j <= *n_dims; j++) 
        {
        	//std::cout<<"i,j "<<i<<" "<<j<<std::endl;	
            in >> (*input_vals)[i][j] ;
        }
	}
}
/*
void write_file(struct options_t*         args,
               	struct prefix_sum_args_t* opts) {
  // Open file
	std::ofstream out;
	out.open(args->out_file, std::ofstream::trunc);

	// Write solution to output file
	for (int i = 0; i < opts->n_vals; ++i) {
		out << opts->output_vals[i] << std::endl;
	}

	out.flush();
	out.close();
	
	// Free memory
	free(opts->input_vals);
	free(opts->output_vals);
}
*/