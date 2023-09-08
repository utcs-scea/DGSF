#include <io.h>

using namespace std;

void read_file_alloc_mem(struct kmean_t *kmean) {

  	
    // declare paramaters
	ifstream in;    
    string line_str;
    float in_val;
    int col, row, cl_idx;
    
    
    // Open file  
	in.open(kmean->in_file);
    
    if(!in) {
        cout << "Cannot open input file" << endl;
        exit(1);
    }
    

    
	// Get num points
	in >> kmean->n_points;

   
    // allocate 2D memory for input points ( n_points x n_dim )
    kmean->input_pts = (float **) malloc(kmean->n_points * sizeof(float*));
    for (row = 0; row < kmean->n_points; row++) {
        
        kmean->input_pts[row] = (float *)malloc( kmean->n_dims * sizeof(float )); 
    
    }
    
    
    // allocate 1D array for cluster labels ( n_points x 1 )
    kmean->cluster_id = (int *) malloc(kmean->n_points * sizeof(int));
    
    // allocate 2D array for centroids ( n_cluster x n_dims)
    kmean->centroids     = (float **) malloc( kmean->n_clusters * sizeof(float *) );
    kmean->old_centroids = (float **) malloc( kmean->n_clusters * sizeof(float *) );
    for ( cl_idx = 0; cl_idx < kmean->n_clusters; cl_idx++) {
        
        kmean->centroids[cl_idx]     = (float *) malloc( kmean->n_dims * sizeof(float));
        kmean->old_centroids[cl_idx] = (float *) malloc( kmean->n_dims * sizeof(float));  
    }
    
    
    // read input_pts
    getline(in, line_str); // initial dummy read to get to line 2
    int line_num;
    
    for( row = 0; row < kmean->n_points; row++ ){

        // read next line and make sure it is not empty
        if (!getline(in, line_str) ) {
            cerr << " unexpected end of file" << endl;
            exit(1);
        }
        
        // read columns
        istringstream iss(line_str);
        
        // read column 0 (row number)
        iss >> line_num;
        if (  line_num != (row+1) ) {
            cerr << "Invalid line number.\n";
            exit(1);
        }
        
        // read columns 1 kmean->n_dim and copy to 2D array
        col = 0; // data points feature or columns
        while ( (iss >> in_val) ) {
            if (col < kmean->n_dims) {
                
                kmean->input_pts[row][col] = in_val;
                //cout << kmean->input_pts[row][col] << ", ";
            }
            col++;
        }
        //cout << "---------------\n";
    }
    
    in.close();
}


void write_file(struct kmean_t * kmean) {
    
    // Open file
	std::ofstream out1, out2;
	out1.open("output/centroids.txt", std::ofstream::trunc);
    out2.open("output/labels.txt", std::ofstream::trunc);
    
	// Write solution to output file
	for (int row = 0; row < kmean->n_clusters; row++) {
        out1 << row << "  ";
        for (int col = 0; col < kmean->n_dims; col++) {
            out1 << kmean->centroids[row][col] << " ";
        }
        out1 << endl;
	}

    for (int row = 0; row < kmean->n_points; row++) {
        out2 << kmean->cluster_id[row] << endl;
    }
    
	out1.flush();  out1.close();
    out2.flush();  out2.close();

    if ( kmean->b_centroid ) print_centroids(kmean->centroids, kmean->n_clusters, kmean->n_dims); 

}

void print_centroids(float **centroids, int n_clusters, int n_dims) {
    
    for (int clusterId = 0; clusterId < n_clusters; clusterId ++) {
        
        printf("%d ", clusterId);
        for (int d = 0; d < n_dims; d++) {
            printf("%lf ", centroids[clusterId][d]);
        }
        printf("\n");
    }
}

void free_mem(struct kmean_t *kmean) {
    
    for( int row = 0; row < kmean->n_points; row++ ) {
        free(kmean->input_pts[row]);
    }
    free(kmean->input_pts);
    
    
    for( int cl_idx = 0; cl_idx < kmean->n_clusters; cl_idx++ ) {
        free(kmean->centroids[cl_idx]);
        free(kmean->old_centroids[cl_idx]);
    }
    free(kmean->centroids);
    free(kmean->old_centroids);
    
    free(kmean->cluster_id);
}