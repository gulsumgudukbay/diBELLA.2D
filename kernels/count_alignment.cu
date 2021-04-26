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


__global__ void count_alignment_kernel(int batch_size, uint64_t local_nnz_count, uint64_t* mattuples0, uint64_t* mattuples1, uint64_t* cks_count,
    uint64_t col_offset,uint64_t row_offset, int ckthr, uint64_t* align_batch, uint64_t* elimi_batch)
{
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    //  printf("%d, %d, %d\n", thread_id,read_idx[thread_id], has_events[thread_id]);
  
      if(i < local_nnz_count)
      {
        uint64_t batch_idx = i/batch_size;
        uint64_t l_row_idx = mattuples0[i];
        uint64_t l_col_idx = mattuples1[i];
        uint64_t g_col_idx = l_col_idx + col_offset;
        uint64_t g_row_idx = l_row_idx + row_offset;
        uint64_t count = cks_count[i];

        assert(l_row_idx >= 0 && l_col_idx >= 0 && g_col_idx >= 0 && g_row_idx >= 0);

				if ((count >= ckthr) 	 	&& 
					(l_col_idx >= l_row_idx) 	&&
					(l_col_idx != l_row_idx  || g_col_idx > g_row_idx))
				{
					//atomicAdd(&align_batch[batch_idx],1);
				}

				if ((l_col_idx >= l_row_idx) &&
					(l_col_idx != l_row_idx || g_col_idx > g_row_idx))
				{
					if (count < ckthr) 
                    {
                     // atomicAdd(&elimi_batch[batch_idx],1);
                    }
				}
      }

}

void count_alignment_cuda(int batch_size, uint64_t local_nnz_count, uint64_t* mattuples0, uint64_t* mattuples1, uint64_t* cks_count,
    uint64_t col_offset,uint64_t row_offset, int ckthr,uint64_t* align_batch, uint64_t* elimi_batch)
{
    int batch_cnt = (local_nnz_count / batch_size) + 1;

    uint64_t* d_mattuples0 = NULL;
    uint64_t* d_mattuples1 = NULL;
    uint64_t* d_cks_count = NULL;
    uint64_t* d_align_batch = NULL;
    uint64_t* d_elimi_batch = NULL;

    

    cudaMalloc((void **)&d_mattuples0, sizeof(uint64_t)*local_nnz_count);
    cudaMalloc((void **)&d_mattuples1, sizeof(uint64_t)*local_nnz_count);
    cudaMalloc((void **)&d_cks_count, sizeof(uint64_t)*local_nnz_count);
    cudaMalloc((void **)&d_align_batch, sizeof(uint64_t)*batch_cnt);
    cudaMalloc((void **)&d_elimi_batch, sizeof(uint64_t)*batch_cnt);

    cudaMemcpy(d_mattuples0,mattuples0,sizeof(uint64_t)*local_nnz_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mattuples1,mattuples1,sizeof(uint64_t)*local_nnz_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_cks_count,cks_count,sizeof(uint64_t)*local_nnz_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_align_batch,align_batch,sizeof(uint64_t)*batch_cnt,cudaMemcpyHostToDevice);
    cudaMemcpy(d_elimi_batch,elimi_batch,sizeof(uint64_t)*batch_cnt,cudaMemcpyHostToDevice);

    uint64_t block_size = 1024;
    uint64_t block_num = (local_nnz_count/block_size)+1;
    count_alignment_kernel<<<block_num, block_size>>>(batch_size, local_nnz_count, d_mattuples0, d_mattuples1, d_cks_count,
        col_offset, row_offset, ckthr, d_align_batch, d_elimi_batch);

    cudaFree(d_mattuples0);
    cudaFree(d_mattuples1);
    cudaFree(d_cks_count);
    cudaFree(d_align_batch);
    cudaFree(d_elimi_batch);
   
}

void test()
{

    std::cout<<"test"<<std::endl;
}