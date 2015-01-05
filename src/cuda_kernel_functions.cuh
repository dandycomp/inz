#ifndef cuda_fuctions
#define cuda_fuctions
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

static const int PATCH_SIZE = 7;// rozwiazac ten problem za posrednictwem parametrow wejsciowych
static const int SUB_IMG = 15;
static const int IMAGE_WIDTH = 133;
static const int IMAGE_HEIGHT = 98;

__global__ void cudaGetBestCorrelateFromArea(int* p_values, int* glsum1,
						int* glsum2, int* glSqSum1, int* glSqSum2,
						int x, int y, float* outp);

__global__ void cudaGetAll_P_FromArea(int *img1, int* img2,
						int x, int y, int* outpPvector);

__global__ void cudaSimpleGetBestCorrelate(int* img2, int x, int y, 
						int* im1_part, float med, float deviat, float *outpCorrelate);

__global__ void cudaGetBestCorrelate_fast(int* img2, int* glSum2, int* glSqSum2,
	int* im1_part, int x, int y,
	int glSum1_val, int glSqSum1_val, float* correlate);


#endif