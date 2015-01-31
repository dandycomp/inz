#ifndef cuda_fuctions
#define cuda_fuctions
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>
#include "parameters.h"

__global__ void cudaFastCorrelation(int* , int* , int , int , float* );
//__global__ void  cudaGetMaxValues(float*, float*, int*);

__global__ void  cudaGetMaxValues(float*, float*, int*);

//__global__ void  cudaGetMaxValues();
#endif