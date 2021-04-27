#ifndef GPU_COUNT_ALIGN_H
#define GPU_COUNT_ALIGN_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void count_alignment_cuda(int batch_size, uint64_t local_nnz_count, uint64_t *mattuples0, uint64_t *mattuples1, uint64_t *cks_count,
                          uint64_t col_offset, uint64_t row_offset, int ckthr, uint64_t *align_batch, uint64_t *elimi_batch);

void test();

#endif