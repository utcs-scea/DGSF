#include <iostream>
#include <argparse.h>
#include <kmean.h>
#include <io.h>
#include <kmean_seq.h>
#include <kmean_thrust.h>
#include <kmean_kernel.h>




using namespace std;

int main(int argc, char **argv) {
    
    cudaEvent_t start, stop;
    float elapsedTime;    
    struct kmean_t kmean;
    // Parse args
    get_opts(argc, argv, &kmean);
    
    

    // read inputs and allocate memory
    read_file_alloc_mem(&kmean);
    




    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    int iter=0;
    float err= 9999999;

    if (kmean.n_rev == 1) {
        //cout << "Sequential Method\n";
        
        kmean_seq *KM_Seq = new kmean_seq(&kmean);
        cudaEventRecord(start, 0);
        KM_Seq->assign_RandomCentroids();

        while ( (iter < kmean.n_max_iter)  && (err > kmean.f_thresh) ) {

            KM_Seq->findNearestCentroids();


            KM_Seq->averageLabeledCentroids();


            err = KM_Seq->calc_Old2NewCentroidsDist();

            //cout << endl << ">>>>>Iter = " << iter << " , err = " << err << endl;

            iter++;
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("%d,%lf\n", iter, elapsedTime/iter);
        free(KM_Seq);

    } else if (kmean.n_rev == 2) {
        //cout << "Thrust Method\n";
        kmean_thrust *KM_Thr = new kmean_thrust(&kmean);

        
        KM_Thr->copy_from_Host_to_Dev();
        cudaEventRecord(start, 0);  
        KM_Thr->assign_RandomCentroids();

        while ( (iter < kmean.n_max_iter)  && (err > kmean.f_thresh) ) {


            KM_Thr->findNearestCentroids();


            KM_Thr->averageLabeledCentroids();


            err = KM_Thr->calc_Old2NewCentroidsDist();


            //cout << endl << ">>>>>Iter = " << iter << " , err = " << err << endl;

            iter++;
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("%d,%lf\n", iter, elapsedTime/iter);
        KM_Thr->copy_from_Dev_to_Host();

        free(KM_Thr);

    } else if ( (kmean.n_rev == 3) || (kmean.n_rev == 4) ){
        //if (kmean.n_rev == 3) 
            //cout << "CUDA Kernel Method\n";
        //else
            //cout << "CUDA Kernel with Shared Memory\n";
        kmean_kernel *KM_Ker = new kmean_kernel(&kmean);


        KM_Ker->copy_from_Host_to_Dev();
        
        cudaEventRecord(start, 0);
        KM_Ker->assign_RandomCentroids();
        while ( (iter < kmean.n_max_iter)  && (err > kmean.f_thresh) ) {

            err = KM_Ker->processKMean();

            //cout << endl << ">>>>>Iter = " << iter << " , err = " << err << endl;

            iter++;
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("%d,%lf\n", iter, elapsedTime/iter );    
        
        KM_Ker->copy_from_Dev_to_Host();

        free(KM_Ker);

    }
    
    write_file(&kmean);
    free_mem(&kmean);
    
    

}

