#ifndef ALLOCATE_STRINGSET_H
#define ALLOCATE_STRINGSET_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//kernel inputs: malloc'd stringSets for GPU, lids, mattuples, col_offset, row_offset, ckthr, dfd
//kernel outputs: seqsh, seqsv, lids
// fill StringSet

void fill_stringset_cuda(uint64_t beg, uint64_t end,uint64_t local_nnz_count, char **seqsh_str, char **seqsv_str, uint64_t *lids, int *mattuples1, int *mattuples2, int *mattuples3, uint64_t row_offset, uint64_t col_offset, int ckthr, char **dfd_col_seq_gpu, char **dfd_row_seq_gpu, uint64_t* align_cnts);

void test1();

#endif



