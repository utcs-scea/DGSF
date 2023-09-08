
#include<math.h>
#include <iostream>
#include <stdio.h>

__device__ float dist(float* a, float* b, int n_dims)
{
	float dis = 0;
    for (int i = 0; i< n_dims; i++)
    {
        //printf("val1 %f val2 %f \n",a[i],b[i]);
        dis = dis + (a[i] - b[i]) * (a[i] - b[i]);
    }
    return dis;
}

__global__ void kmeans_dist_assgn(float* d_input_vals,float* d_clusters,int* d_cluster_id,int n_vals,int n_dims,int n_clusters, int* d_cluster_size, float* d_sum_clusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_vals)
        return;
    
    extern __shared__ float s_clusters[];
    
    //printf("launch %d %d \n", threadIdx.x, n_clusters);
    
    if(threadIdx.x < n_clusters)
    {
            //printf("launch \n");
            for(int j = 0; j< n_dims;j++)
            {
                s_clusters[threadIdx.x + j] = d_clusters[threadIdx.x + j];
                //printf("%f %f ",s_clusters[threadIdx.x + j],d_clusters[threadIdx.x + j]);
            }
        
        
    }
    __syncthreads();
    
        float min_dis = 0;
        int min_idx = 0;
        
        min_dis = dist(&d_input_vals[idx*n_dims ], s_clusters, n_dims);

        for (int i = 1; i < n_clusters; i++)
        {
            float dis = dist(&d_input_vals[idx*n_dims ], &s_clusters[n_dims* i], n_dims);
            //printf("thread %d index %d input %f, cluster %f %f\n", threadIdx.x, idx, dis,d_input_vals[idx*n_dims],d_input_vals[n_dims*idx+1] );
            if (dis< min_dis)
            {
                //printf("sud \n");
                min_dis = dis;
                min_idx = i; 
            }
        }
        for(int i = 0; i< n_dims; i++)
        {
            //printf("check idx %d %d %d %f %f %f %f \n",i, idx,d_cluster_id[idx]*n_dims, d_clusters[d_cluster_id[idx]*n_dims],d_clusters[d_cluster_id[idx]*n_dims+1], d_input_vals[idx*n_dims], d_input_vals[idx*n_dims +1]);
            //d_clusters[d_cluster_id[idx]*n_dims +i ] = d_clusters[d_cluster_id[idx]*n_dims +i ] + d_input_vals[idx*n_dims + i];
            atomicAdd(&d_sum_clusters[min_idx*n_dims +i ], d_input_vals[idx*n_dims + i ]);
        }
        atomicAdd(&d_cluster_size[min_idx],1);
        //if(idx==59999)
        //printf("lol thread_id %d, min_idx %d, min_dis %f %f %f \n", idx, min_idx, min_dis, d_input_vals[idx*n_dims], d_clusters[min_idx*n_dims]);
        d_cluster_id[idx] = min_idx;
    
    
}

__global__ void kmeans_new_centroid(float* d_input_vals,float* d_clusters,int* d_cluster_id,int n_vals,int n_dims,int n_clusters, int* d_cluster_size, float* d_sum_clusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //float* up_cluster[n_clusters];
    //printf("check %d %d \n", blockIdx.x,threadIdx.x);
    //printf("cluster size %d \n", d_cluster_size[0]);
    if(idx < n_vals)
    {
        /*
        for(int i = 0; i< n_dims; i++)
        {
            //printf("check idx %d %d %d %f %f %f %f \n",i, idx,d_cluster_id[idx]*n_dims, d_clusters[d_cluster_id[idx]*n_dims],d_clusters[d_cluster_id[idx]*n_dims+1], d_input_vals[idx*n_dims], d_input_vals[idx*n_dims +1]);
            //d_clusters[d_cluster_id[idx]*n_dims +i ] = d_clusters[d_cluster_id[idx]*n_dims +i ] + d_input_vals[idx*n_dims + i];
            atomicAdd(&d_clusters[d_cluster_id[idx]*n_dims +i ], d_input_vals[idx*n_dims + i]);
        }
        atomicAdd(&d_cluster_size[d_cluster_id[idx]],1);
        */
    }
    
    
    /*
    __syncthreads();
    if(idx == 0)
        printf("cluster size %d", d_cluster_size[0]);
    
    if(idx < n_clusters)
    {
        for(int i = 0; i< n_dims; i++)
        {
            d_clusters[idx*n_dims + i] = d_clusters[idx*n_dims +i]/d_cluster_size[i];   
        }
    }
    __syncthreads();
    if(idx < n_vals)
    {
    printf("check idx %d %d %f %f \n", idx,d_cluster_id[idx], d_clusters[d_cluster_id[idx]*n_dims],d_clusters[d_cluster_id[idx]*n_dims+1]);
    }
    */
    int size = max(d_cluster_size[idx],1);
    for(int i = 0; i< n_dims; i++)
    {
        d_clusters[idx*n_dims+i] = d_sum_clusters[idx*n_dims+i]/size;
    }
    
    
}



void kmeans_kernel_launch(int n_vals, int n_dims,int* cluster_id,float* input_vals, int max_iter,double threshold,float* clusters, int n_clusters)
{
    cudaSetDevice(0);
    
    int thr_per_blk = 128;
    int nblocks = int(ceil(float(n_vals) / thr_per_blk));
    //printf("blocks %d", nblocks);
    float* d_input_vals;
    float* d_clusters;
    int* d_cluster_id;
    int* d_cluster_size;
    float* d_sum_clusters;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
  
    
    cudaEventRecord(start);
    cudaMalloc(&d_input_vals,n_vals*n_dims*sizeof(float));
    cudaMalloc(&d_clusters,n_clusters*n_dims*sizeof(float));
    cudaMalloc(&d_cluster_id,n_vals*sizeof(int));
    cudaMalloc(&d_cluster_size,n_clusters*sizeof(int));
    cudaMalloc(&d_sum_clusters,n_clusters*n_dims*sizeof(float));
    
    cudaMemcpy(d_input_vals, input_vals, n_vals*n_dims*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusters, clusters, n_clusters*n_dims*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_cluster_id, cluster_id, n_vals*sizeof(int), cudaMemcpyHostToDevice);
    
    float total_time = 0;
    int j;
    
    for(j = 0; j< max_iter; j++)
    {
        
        cudaMemset(d_sum_clusters,0.0, n_dims*n_clusters*sizeof(float));
        cudaMemset(d_cluster_size,0, n_clusters*sizeof(int));
        cudaMemset(d_cluster_id,0, n_vals*sizeof(int));
        //std::cout<<"kernel launch"<<std::endl;
        //std::cout<<"val_old "<<nblocks<<" "<<thr_per_blk<<std::endl;
        
        kmeans_dist_assgn<<<nblocks,thr_per_blk,n_clusters * n_dims>>>(d_input_vals,d_clusters,d_cluster_id,n_vals,n_dims,n_clusters,d_cluster_size, d_sum_clusters);        
    
    
        //std::cout<<"kernel end"<<std::endl;

        //kmeans_new_centroid<<<nblocks,thr_per_blk>>>(d_input_vals,d_clusters,d_cluster_id,n_vals,n_dims,n_clusters, d_cluster_size);        
        kmeans_new_centroid<<<1,n_clusters>>>(d_input_vals,d_clusters,d_cluster_id,n_vals,n_dims,n_clusters, d_cluster_size,d_sum_clusters);        
    
        
        
        cudaDeviceSynchronize();
        //std::cout<<"Iteration"<< j<<std::endl;
    
        //float milliseconds = 0;
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //printf("\n");
        //total_time += milliseconds;
    }
    //cudaEventSynchronize(stop);
    
    
    cudaMemcpy(clusters, d_clusters, n_clusters*n_dims*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_id, d_cluster_id, n_vals*sizeof(int), cudaMemcpyDeviceToHost);
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);
    printf("%d,%lf\n", j, total_time/j);
    
    cudaFree(d_input_vals);
    cudaFree(d_clusters);
    cudaFree(d_cluster_id);
    cudaFree(d_sum_clusters);
    cudaFree(d_cluster_size);
}
