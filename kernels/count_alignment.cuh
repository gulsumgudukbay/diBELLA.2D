#ifndef GPU_COUNT_ALIGN_H
#define GPU_COUNT_ALIGN_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void count_alignment_cuda(int batch_size, int local_nnz_count, int* mattuples0, int* mattuples1, int* cks_count,
    int col_offset,int row_offset, int ckthr, int* align_batch);

void test();

#endif