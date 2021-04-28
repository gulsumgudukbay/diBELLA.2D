#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <fstream>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include "allocate_stringset.cuh"
__global__ void fill_stringset_kernel(uint64_t beg, uint64_t end, uint64_t local_nnz_count, char **seqsh_str, char **seqsv_str, uint64_t *lids, uint64_t *mattuples1, uint64_t *mattuples2, uint64_t* mattuples3, uint64_t row_offset, uint64_t col_offset, int ckthr, char **dfd_col_seq_gpu, char **dfd_row_seq_gpu, uint64_t* align_cnts)
{
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t algn_idx = align_cnts[i];

    if (i < end-beg+1)
    {
        uint64_t l_row_idx = mattuples1[i];
        uint64_t l_col_idx = mattuples2[i];
        uint64_t g_col_idx = l_col_idx + col_offset;
        uint64_t g_row_idx = l_row_idx + row_offset;
        uint64_t cks_count = mattuples3[i];

        assert(l_row_idx >= 0 && l_col_idx >= 0 && g_col_idx >= 0 && g_row_idx >= 0);

        if ((cks_count >= ckthr) &&
            (l_col_idx >= l_row_idx) &&
            (l_col_idx != l_row_idx || g_col_idx > g_row_idx))
        {
            seqsh_str[algn_idx] = dfd_col_seq_gpu[l_col_idx];
            seqsv_str[algn_idx] = dfd_row_seq_gpu[l_row_idx];
            printf( "test_stringset assigned %d --> %s and %s \n" , i, seqsh_str[algn_idx], seqsv_str[algn_idx]);

            lids[algn_idx] = i;
            // ++algn_idx;
        }
    }
}

void fill_stringset_cuda(uint64_t beg, uint64_t end,uint64_t local_nnz_count, char **seqsh_str, char **seqsv_str, uint64_t *lids, uint64_t *mattuples1, uint64_t *mattuples2, uint64_t *mattuples3, uint64_t row_offset, uint64_t col_offset, int ckthr, char **dfd_col_seq_gpu, char **dfd_row_seq_gpu, uint64_t* align_cnts)
{
    uint64_t *d_mattuples1 = NULL;
    uint64_t *d_mattuples2 = NULL;
    uint64_t *d_mattuples3 = NULL;
    char **d_seqsh_str = NULL;
    char **d_seqsv_str = NULL;
    char ** d_dfd_row_seq_gpu = NULL;
    char ** d_dfd_col_seq_gpu = NULL;
    uint64_t *d_lids = NULL;
    uint64_t no_len = end-beg; //CHECK
    uint64_t block_size = 1024;
    uint64_t block_num = (no_len / block_size) + 1;

    uint64_t numThreads = 1000;

    cudaMalloc((void **)&d_mattuples1, sizeof(uint64_t) * local_nnz_count);
    cudaMalloc((void **)&d_mattuples2, sizeof(uint64_t) * local_nnz_count);
    cudaMalloc((void **)&d_mattuples3, sizeof(uint64_t) * local_nnz_count);

    cudaMalloc((void **)&d_seqsh_str, sizeof(char*)*align_cnts[numThreads]);
    cudaMalloc((void **)&d_seqsv_str, sizeof(char*)*align_cnts[numThreads]);
    cudaMalloc((void **)&d_lids, sizeof(char*)*align_cnts[numThreads]);

    cudaMalloc((void **)&d_dfd_col_seq_gpu, sizeof(char*)*align_cnts[numThreads]);
    cudaMalloc((void **)&d_dfd_row_seq_gpu, sizeof(char*)*align_cnts[numThreads]);

    cudaMemcpy(d_mattuples1, mattuples1, sizeof(uint64_t) * local_nnz_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mattuples2, mattuples2, sizeof(uint64_t) * local_nnz_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mattuples3, mattuples3, sizeof(uint64_t) * local_nnz_count, cudaMemcpyHostToDevice);

    cudaMemcpy(d_seqsh_str, seqsh_str, sizeof(char*)*align_cnts[numThreads], cudaMemcpyHostToDevice);
    cudaMemcpy(d_seqsv_str, seqsv_str, sizeof(char*)*align_cnts[numThreads], cudaMemcpyHostToDevice);
    cudaMemcpy(d_lids, lids, sizeof(char*)*align_cnts[numThreads], cudaMemcpyHostToDevice);

    cudaMemcpy(d_dfd_col_seq_gpu, dfd_col_seq_gpu, sizeof(char*)*align_cnts[numThreads], cudaMemcpyHostToDevice);
    cudaMemcpy(d_dfd_row_seq_gpu, dfd_row_seq_gpu, sizeof(char*)*align_cnts[numThreads], cudaMemcpyHostToDevice);


    fill_stringset_kernel<<<block_num, block_size>>>(beg, end, local_nnz_count, d_seqsh_str, d_seqsv_str, d_lids, d_mattuples1, d_mattuples2, d_mattuples3, row_offset, col_offset, ckthr, d_dfd_col_seq_gpu, d_dfd_row_seq_gpu, align_cnts);

    cudaFree(d_mattuples1);
    cudaFree(d_mattuples2);
    cudaFree(d_mattuples3);
    cudaFree(d_seqsh_str);
    cudaFree(d_seqsv_str);
    cudaFree(d_lids);
    cudaFree(d_dfd_col_seq_gpu);
    cudaFree(d_dfd_row_seq_gpu);
}


void test1()
{
    std::cout << "test" << std::endl;

}
