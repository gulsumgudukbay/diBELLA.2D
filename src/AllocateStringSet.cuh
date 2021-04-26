#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

//kernel inputs: malloc'd stringSets for GPU, lids, mattuples, col_offset, row_offset, ckthr, dfd
//kernel outputs: seqsh, seqsv, lids
// fill StringSet

__global__ void fillStringSet(char** seqsh_str, char** seqsv_str, uint64_t *lids, int* mattuples1, int* mattuples2, int* mattuples3, uint64_t row_offset, uint64_t col_offset, int ckthr);