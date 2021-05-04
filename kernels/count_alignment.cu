#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <fstream>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include "count_alignment.cuh"


__global__ void count_alignment_kernel(int batch_size, int local_nnz_count, int* mattuples0, int* mattuples1, int* cks_count,
    int col_offset, int row_offset, int ckthr, int* align_batch)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //  printf("%d, %d, %d\n", thread_id,read_idx[thread_id], has_events[thread_id]);
  
      if(i < local_nnz_count)
      {
        int batch_idx = i/batch_size;
        int l_row_idx = mattuples0[i];
        int l_col_idx = mattuples1[i];
        int g_col_idx = l_col_idx + col_offset;
        int g_row_idx = l_row_idx + row_offset;
        int count = cks_count[i];

 //       assert(l_row_idx >= 0 && l_col_idx >= 0 && g_col_idx >= 0 && g_row_idx >= 0);

				if ((count >= ckthr) 	 	&& 
					(l_col_idx >= l_row_idx) 	&&
					(l_col_idx != l_row_idx  || g_col_idx > g_row_idx))
				{
					atomicAdd(&align_batch[batch_idx],1);
					//align_batch[batch_idx]++;
				}

	/*			if ((l_col_idx >= l_row_idx) &&
					(l_col_idx != l_row_idx || g_col_idx > g_row_idx))
				{
					if (count < ckthr) 
                    {
                      atomicAdd(&elimi_batch[batch_idx],1);
			//elimi_batch[batch_idx]++;
                    }
				}*/
      }
 
}

void count_alignment_cuda(int batch_size, int local_nnz_count, int* mattuples0, int* mattuples1, int* cks_count,
    int col_offset, int row_offset, int ckthr,int* align_batch)
{
    
    int batch_cnt = (local_nnz_count / batch_size) + 1;

    int* d_mattuples0 = NULL;
    int* d_mattuples1 = NULL;
    int* d_cks_count = NULL;
    int* d_align_batch = NULL;
    int* d_elimi_batch = NULL;
 

    cudaMalloc((void **)&d_mattuples0, sizeof(int)*local_nnz_count);
    cudaMalloc((void **)&d_mattuples1, sizeof(int)*local_nnz_count);
    cudaMalloc((void **)&d_cks_count, sizeof(int)*local_nnz_count);
    cudaMalloc((void **)&d_align_batch, sizeof(int)*batch_cnt);
  // cudaMalloc((void **)&d_elimi_batch, sizeof(int)*batch_cnt);

    cudaMemcpy(d_mattuples0,mattuples0,sizeof(int)*local_nnz_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mattuples1,mattuples1,sizeof(int)*local_nnz_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_cks_count,cks_count,sizeof(int)*local_nnz_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_align_batch,align_batch,sizeof(int)*batch_cnt,cudaMemcpyHostToDevice);
   // cudaMemcpy(d_elimi_batch,elimi_batch,sizeof(int)*batch_cnt,cudaMemcpyHostToDevice);

    int block_size = 512;
    int block_num = (local_nnz_count/block_size)+1;
    count_alignment_kernel<<<block_num, block_size>>>(batch_size, local_nnz_count, d_mattuples0, d_mattuples1, d_cks_count,
<<<<<<< HEAD
        col_offset, row_offset, ckthr, d_align_batch);
=======
                                                      col_offset, row_offset, ckthr, d_align_batch, d_elimi_batch);
>>>>>>> e675383b7b4be13537ab4b835452eb74f40bc8e5

    cudaMemcpy(align_batch,d_align_batch, sizeof(int)*batch_cnt, cudaMemcpyDeviceToHost);
//    for(int i=0;i<batch_cnt;i++)
  //     std::cout<<"batch "<<i<<" align_batch"<<align_batch[i]<<std::endl;

<<<<<<< HEAD

   
=======
    cudaFree(d_mattuples0);
    cudaFree(d_mattuples1);
    cudaFree(d_cks_count);
    cudaFree(d_align_batch);
    cudaFree(d_elimi_batch);
>>>>>>> e675383b7b4be13537ab4b835452eb74f40bc8e5
}

void test()
{

    std::cout<<"test"<<std::endl;
}
