#include "cuda_code.cuh"
#include "utils.hpp"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#ifndef __CUDACC__  
	#define __CUDACC__
#endif

__global__ void helloWorld(uchar* str)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	str[idx] += idx;
}

__global__ void cudaKernel(uchar * tab, int * outp)
{
	int i = threadIdx.x;
	outp[i] = (int)tab[i];
}

__global__ void getP(uchar* img1, uchar* img2,
					int x1, int y1, int x2, int y2, int patch,
					int width, int* outpPval)
{
	int thread = threadIdx.x;
	int dispX = thread % patch - patch / 2;// displacement X
	int dispY = thread / patch - patch / 2;

	int index1 = (dispX + x1) + width * (dispY + y1);
	int index2 = (dispX + x2) + width * (dispY + y2);
	int t1 = img1[index1];
	int t2 = img1[index2];
	outpPval[thread] = t1 * t2;
}

__global__ void getPMat(uchar* img1, uchar* img2, int x1, int y1,
	int x2, int y2, int patch, int subImg, int imgWidth, int* outpVal)
{
#if 0
	int idx = blockIdx.x;
	int idy = blockIdx.y;


	int dispX = idx - patch / 2 - 1;
	int dispY = idy - patch / 2 - 1;

	int sizeOfArea = subImg - patch + 1;
	int size = patch * patch;
	//__shared__ int *cudaOutp;
	//cudaMalloc(&cudaOutp, size*sizeof(int));
	
	int x1_ = dispX + x1;
	int y1_ = dispY + y1;
	int x2_ = dispX + x2;
	int y2_ = dispY + y2;

//	getP << <1, size >> >(img1, img2,
//		x1_, y1_, x2_, y2_, patch,
//		imgWidth, cudaOutp);
	//__syncthreads();

//	int sum = 0;
//	for (int i = 0; i < patch; i++)
//		 sum += cudaOutp[i];

	outpVal[idx + sizeOfArea*idy] = sum;
	//cudaFree(cudaOutp);
#endif
}

 void CUDAGlobalPmat::deviceInfo(){
	 int devCount;
	 cudaGetDeviceCount(&devCount);
	 printf("CUDA Device Query...\n");
	 printf("There are %d CUDA devices.\n", devCount);

	 cudaDeviceProp devProp;
	 for (int i = 0; i < devCount; ++i)
	 {
		 cudaGetDeviceProperties(&devProp, i);
		 printf("Major revision number:         %d\n", devProp.major);
		 printf("Minor revision number:         %d\n", devProp.minor);
		 printf("Name:                          %s\n", devProp.name);
		 printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
		 printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
		 printf("Total registers per block:     %d\n", devProp.regsPerBlock);
		 printf("Warp size:                     %d\n", devProp.warpSize);
		 printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
		 printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
		 for (int i = 0; i < 3; ++i)
			 printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
		 for (int i = 0; i < 3; ++i)
			 printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
		 printf("Clock rate:                    %d\n", devProp.clockRate);
		 printf("Total constant memory:         %lu\n", devProp.totalConstMem);
		 printf("Texture alignment:             %lu\n", devProp.textureAlignment);
		 printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
		 printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
		 printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	 }
 }

CUDAGlobalPmat::CUDAGlobalPmat(CUDAGlobalPmat& pm){

    pm.m_CUDAGlobalPmat.copyTo(m_CUDAGlobalPmat);//pozniej mozna pomyslec nad referencjami
	//memcpy(m_CUDAGlobalPmat, pm.m_CUDAGlobalPmat, sizeof(pm.m_CUDAGlobalPmat));
    m_subImg = pm.m_subImg;
    m_patch  = pm.m_patch;
}

CUDAGlobalPmat::CUDAGlobalPmat(){
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		this->~CUDAGlobalPmat();
	}
}

void CUDAGlobalPmat::setParameters(cv::Mat& img1,
	cv::Mat& img2, int subImg, int patch)
{
	cudaError_t cudaStatus;
    //NIE MA ZADNYCH SPRAWDZEN, TRZEBA POPRAWIC
    m_patch = patch;
    m_subImg = subImg;
	m_cols = img1.cols;

	int nrOfRows = img1.rows 
					- m_subImg + 1; // rows in output image
	int nrOfCols = img1.cols 
					- m_subImg + 1; // cols in output image
	int pixelVal = nrOfRows * nrOfCols;

	m_CUDAGlobalPmat = Mat::zeros(nrOfRows, nrOfCols, CV_32F);
	
	m_arrayImg1 = new uchar[pixelVal];// array-like Mat
	m_arrayImg2 = new uchar[pixelVal];// array-like Mat

	int counter = 0;
	for (int i = 0; i < nrOfRows; i++)
		for (int j = 0; j < nrOfCols; j++){
		counter = j + i*nrOfCols;
		m_arrayImg1[counter] = img1.at<uchar>(i, j);
		m_arrayImg2[counter] = img2.at<uchar>(i, j);
		}

//	cudaStatus = cudaMalloc((void**)&m_cudaImg1, pixelVal*sizeof(uchar));
	cudaStatus = cudaMalloc(&m_cudaImg1, pixelVal*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		this->~CUDAGlobalPmat();
	}

//	cudaStatus = cudaMalloc((void**)&m_cudaImg2, pixelVal*sizeof(uchar));
	cudaStatus = cudaMalloc(&m_cudaImg2, pixelVal*sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		this->~CUDAGlobalPmat();
	}

	cudaStatus = cudaMemcpy(m_cudaImg1, m_arrayImg1, pixelVal*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		this->~CUDAGlobalPmat();
	}

	cudaStatus = cudaMemcpy(m_cudaImg2, m_arrayImg2, pixelVal*sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		this->~CUDAGlobalPmat();
	}
}

void CUDAGlobalPmat::run()
{
	//deviceInfo();
	cudaError_t cudaStatus;


//	getP <<<1, size >> >(m_cudaImg1, m_cudaImg2,
//		100, 100, 100, 100, m_patch,
//		m_cols, cudaOutp);

	//int p = getPFromImages(m_cudaImg1, m_cudaImg2,
	//	100, 100, 100, 100, m_patch,
	//	m_cols);

	//cout << " Wartosc P wynosi = " << p << endl;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		this->~CUDAGlobalPmat();
	}
	/*
	cudaStatus = cudaMemcpy(outp, cudaOutp, size*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		this->~CUDAGlobalPmat();
	}

	cudaFree(cudaOutp);
	*/
}
CUDAGlobalPmat :: ~CUDAGlobalPmat()
{
	free(m_arrayImg1);
	free(m_arrayImg2);
	cudaFree(m_cudaImg1);
	cudaFree(m_cudaImg2);
}

Mat CUDAGlobalPmat::getPmat(Point, Point){

	int pMatsize = m_subImg - m_patch + 1;
	Mat pMat = Mat::zeros(pMatsize, pMatsize, CV_32F);



	return pMat;
}
int CUDAGlobalPmat::getPFromImages(uchar* img1, uchar* img2,
	int x1, int y1, int x2, int y2, int patch,
	int width)
{
	int size = m_patch * m_patch;
	int *cudaOutp;
	int *outp = (int*)malloc(size*sizeof(int));

	cudaMalloc(&cudaOutp, size*sizeof(int));

	//getP << <1, size >> >(img1, img2,
	//	x1, y1, x2, y2, patch,
	//	width, cudaOutp);

	cudaMemcpy(outp, cudaOutp, size*sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < patch*patch; i++)
	{
		int tmp = outp[i];
		cout << "outp = " << tmp << endl;
		sum += tmp;
	}
	cudaFree(cudaOutp);
	return sum;
}