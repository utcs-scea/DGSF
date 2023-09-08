#include "kmeans_seq.h"

double distance(int n_dims, double* input_vals, double* clusters)
{
    double dist = 0;
    for (int i = 0; i <n_dims; i++)
    {
        dist = dist + ((input_vals[i] - clusters[i]) * (input_vals[i] - clusters[i]));
    }
    return pow(dist, 1/double(n_dims));
}

void kmeans_seq(int n_vals,int n_dims,int* cluster_id, double** input_vals, int max_iter, double threshold, double*** clusters, int n_clusters )
{
    //cluster_id = (int*) malloc((n_vals) * sizeof(int));
    //std::cout<<max_iter<<" "<<n_dims<<std::endl;
    //bool update_flag = true;
    auto start = std::chrono::high_resolution_clock::now();
    float total_time = 0;
    int i;
    for (i = 0; i < max_iter; i++)
    {
        bool update_flag = false;
        for (int j = 0; j< n_vals; j++)
        {
            //cluster_id[j] = (int*) malloc((n_vals) * sizeof(int));
            double dist=distance(n_dims,input_vals[j],(*clusters)[0]) ;
            cluster_id[j]=0;
            for (int k = 1; k < n_clusters; k++)
            {
                //std::cout<<dist<<" "<<cluster_id[j]<<" "<<distance(n_dims,input_vals[j],(*clusters)[k])<<std::endl;
                
                if(dist>distance(n_dims,input_vals[j],(*clusters)[k]))
                {
                    dist = distance(n_dims,input_vals[j],(*clusters)[k]);
                    cluster_id[j] = k;
                }
                
            }
            //std::cout<<cluster_id[j]<<" index "<<j<<std::endl;
            //std::cout<<std::endl;
        }
        double* total;
        total = (double*) malloc((n_dims) * sizeof(double));
        for (int k = 0; k < n_clusters; k++)
        {
            int count=0;
            
            bool flag=false;
            
            for (int i = 0; i< n_dims; i++)
            {
                total[i] = 0;
            }
            for (int j = 0; j< n_vals; j++)
            {
                //count =0;
                if(cluster_id[j]==k)
                {
                    flag = true;
                    for (int i = 0; i< n_dims; i++)
                    {
                        total[i] = total[i] + input_vals[j][i];
                        //std::cout<<k<<" "<<input_vals[j][i]<<" ";
                    }
                    
                    count++;
                }
            }
            //std::cout<<"sud"<<k<<" "<<total[0]<<" "<<std::endl;
            
            if(flag)
            {
                
                for (int i = 0; i< n_dims; i++)
                {
                    total[i] = total[i]/count;
                    //std::cout<<total[i]<<" total ";
                }
                if(abs((*clusters)[k][i] - total[i]) < threshold)
                {
                    continue;
                }
                update_flag = true;
                for (int i = 0; i< n_dims; i++)
                {
                    (*clusters)[k][i] = total[i] ;
                    //std::cout<<(*clusters)[k][i]<<" ";
                }
            }
            //std::cout<<std::endl;
        }
        if(update_flag == false)
            break;
        //std::cout<<std::endl;
        
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> mili = end - start;
    total_time += mili.count();
    printf("%d,%lf\n", i, total_time/(i+1));
    //std::cout<<"iter"<<i<<std::endl;
}