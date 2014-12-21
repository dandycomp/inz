#include "cuda_thrust.cuh"
#include "utils.hpp"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#if 0
__global__ void helloWorld(char* str)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	str[idx] += idx;
}

__global__ void cudaKernel(char * tab, int * outp)
{
	int i = threadIdx.x;
	outp[i] = (int)tab[i];
}

__global__ void getP(char* img1, char* img2,
	int x1, int y1, int x2, int y2, int patch,
	int width, int& outpPval)
{
	int x_ = threadIdx.x - patch / 2;
	int y_ = blockIdx.x - patch / 2;
	//printf("\n(%d,%d)", (x_+ x1), (y_+y1));
	int index1 = (x_ + x1) + width * (y_ + y1);
	int index2 = (x_ + x2) + width * (y_ + y2);
	int outp = (int)img1[index1] * (int)img2[index2];
	__syncthreads();
	//printf("\nIndex1 = %d, index2 = %d, outp = %d", (int)img1[index1], (int)img2[index2], outp);// , outpVal = %d", index1, index2, outp, outpPval);// , outp = %d, outpVal = %d", outp, outpPval);
	outpPval += outp;
}

void hello()
{
	int i;
	char str[] = "Hello!";
	for (i = 0; i < 6; i++)
		str[i] -= i;
	char *d_str;
	int size = sizeof(str);
	cudaMalloc((void**)&d_str, size);
	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);
	dim3 dimGrid(2);
	dim3 dimBlock(3);
	helloWorld << <dimGrid, dimBlock >> >(d_str);
	cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);
	cudaFree(d_str);
	printf("\n%s\n", str);
}

CUDAGlobalPmat::CUDAGlobalPmat(CUDAGlobalPmat& pm){

	pm.m_CUDAGlobalPmat.copyTo(m_CUDAGlobalPmat);//pozniej mozna pomyslec nad referencjami
	//memcpy(m_CUDAGlobalPmat, pm.m_CUDAGlobalPmat, sizeof(pm.m_CUDAGlobalPmat));
	m_subImg = pm.m_subImg;
	m_patch = pm.m_patch;
	m_pnt = pm.m_pnt;
}

void CUDAGlobalPmat::info()
{
	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;
	cout << "Thrust v" << major << "." << minor << endl;

	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printf("Major revision number:         %d\n", devProp.major);
		printf("Minor revision number:         %d\n", devProp.minor);
		printf("Name:                          %s\n", devProp.name);
		printf("Total global memory:           %u\n", devProp.totalGlobalMem);
		printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
		printf("Total registers per block:     %d\n", devProp.regsPerBlock);
		printf("Warp size:                     %d\n", devProp.warpSize);
		printf("Maximum memory pitch:          %u\n", devProp.memPitch);
		printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
		for (int i = 0; i < 3; ++i)
			printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
		for (int i = 0; i < 3; ++i)
			printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
		printf("Clock rate:                    %d\n", devProp.clockRate);
		printf("Total constant memory:         %u\n", devProp.totalConstMem);
		printf("Texture alignment:             %u\n", devProp.textureAlignment);
		printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
		printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
		printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
		printf("Maximum resident threads per multiprocessor:     %d\n", devProp.maxThreadsPerMultiProcessor);
	}
}

void CUDAGlobalPmat::setParameters(Point pnt, cv::Mat& img1,
	cv::Mat& img2, int subImg, int patch)
{

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	//NIE MA ZADNYCH SPRAWDZEN, TRZEBA POPRAWIC
	m_pnt = pnt;
	m_patch = patch;
	m_subImg = subImg;
	m_cols = img1.cols;

	int nrOfRows = img1.rows
		- m_subImg + 1; // rows in output image
	int nrOfCols = img1.cols
		- m_subImg + 1; // cols in output image
	int pixelVal = nrOfRows * nrOfCols;

	m_CUDAGlobalPmat = Mat::zeros(nrOfRows, nrOfCols, CV_32F);

	m_arrayImg1 = new char[pixelVal];// array-like Mat
	m_arrayImg2 = new char[pixelVal];// array-like Mat

	int counter = 0;
	for (int i = 0; i < nrOfRows; i++)
		for (int j = 0; j < nrOfCols; j++){
		counter = j + i*nrOfCols;
		m_arrayImg1[counter] = img1.at<uchar>(i, j);
		m_arrayImg2[counter] = img2.at<uchar>(i, j);
		}

	cudaStatus = cudaMalloc((void**)&m_cudaImg1, pixelVal*sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&m_cudaImg2, pixelVal*sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(m_cudaImg1, m_arrayImg1, pixelVal*sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		//		goto Error;
	}

	cudaStatus = cudaMemcpy(m_cudaImg2, m_arrayImg2, pixelVal*sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		//goto Error;
	}

}

void CUDAGlobalPmat::run()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	const int size = m_patch * m_patch;
	//int *outp = new int [size];
	//for (int i = 0; i < size; i++)
	//	outp[i] = 0;

	int outp = 0;
	getP << <m_patch, m_patch >> >(m_cudaImg1, m_cudaImg2,
		100, 100, 100, 100, m_patch,
		m_cols, outp);
	cudaDeviceSynchronize();
	//cvWaitKey(100);
	cout << " Output value of P = " << outp;
#if 0
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int nrOfCols = (*m_img1).cols;
	int nrOfRows = (*m_img1).rows;
	const int size = 20;

	char *image = new char[nrOfRows * nrOfCols];
	int outp[size] = { 0 };
	int counter = 0;

	char *cudaImage = NULL;
	int  *cudaOutp = NULL;

	//cout << " Firstly: " << endl;
	for (int i = 0; i < nrOfRows; i++)
		for (int j = 0; j < nrOfCols; j++){
		counter = j + i*nrOfCols;
		image[counter] = (*m_img1).at<uchar>(i, j);
		}

	cudaStatus = cudaMalloc((void**)&cudaImage, nrOfCols*nrOfRows*sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&cudaOutp, size*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cudaImage, image, nrOfCols*nrOfRows*sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaKernel << <1, size >> > (cudaImage, cudaOutp);

	cudaStatus = cudaMemcpy(outp, cudaOutp, size*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	for (int i = 0; i < size; i++)
	{
		cout << i << ". img = " << (int)image[i] << " outp = " << outp[i] << endl;
	}

Error:
	cudaFree(cudaOutp);
	cudaFree(cudaImage);
#endif
}

/*
int CUDAGlobalPmat::getPFromImages(Point p1, Point p2)
{
int p_val = 0;
for(int i = -m_patch/2; i <= m_patch/2; i++)
for(int j = -m_patch/2; j <= m_patch/2; j++)
{
int first  = (*m_img1).at<uchar>(p1.y+j, p1.x+i);
int second = (*m_img2).at<uchar>(p2.y+j, p2.x+i);
p_val += first * second;
}
return p_val;
}
*/
CUDAGlobalPmat :: ~CUDAGlobalPmat()
{
	cudaFree(m_cudaImg1);
	cudaFree(m_cudaImg2);
}

#endif

using thrust::device_vector;
using thrust::host_vector;

host_vector<int> CUDAGlobalPmat::matToHostVector(Mat& img)
{	
	int tempVal = 0;
	int counter = 0;
	host_vector<int> vec(img.rows*img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			tempVal = img.at<uchar>(i,j);
			vec[counter] = tempVal;
			counter++;
		}
	return vec;
}

CUDAGlobalPmat::~CUDAGlobalPmat()
{
	m_imgVec1.clear();
	m_imgVec2.clear();
	m_PGlMat.clear();
}

__global__ void getPvec(device_vector<int>* img1, device_vector<int>* img2,
					int x1, int y1, int x2, int y2, int patch, int width, 
					device_vector<int>* output)
{
	// output size == patch * patch
	int thr = threadIdx.x;
	int xDisplacement = thr % patch - patch / 2;
	int yDisplacement = thr / patch - patch / 2;

	int index1 = x1 + xDisplacement + (y1 + yDisplacement) * width;
	int index2 = x2 + xDisplacement + (y2 + yDisplacement) * width;
	
	*output[thr] = *img1[index1] * *img2[index2];
}

void CUDAGlobalPmat::setParameters(Mat& img1, Mat& img2, int subImg, int patchSize)
{
//	sprawdzic czy obraz szrosciowy
//	m_img1 = &img1;
//	m_img2 = &img2;
	m_subImg = subImg;
	m_patch = patchSize;
	m_sizeImg = img1.size();

	m_imgVec1.clear();
	m_imgVec2.clear();

	m_imgVec1 = matToHostVector(img1);
	m_imgVec2 = matToHostVector(img2);

	m_PGlMat.clear();
}

void CUDAGlobalPmat::run(){

	cout << "Pixel Value of [100,100] "
		<< getPixVal(m_imgVec1, Point(102,100)) << endl;
}

int CUDAGlobalPmat::getIndex(Point pnt){
	return pnt.x + pnt.y*m_sizeImg.width;
}
int CUDAGlobalPmat::getIndex(int x, int y){
	return x + y*m_sizeImg.width;
}
int CUDAGlobalPmat::getPixVal(host_vector<int>& img, Point pnt)
{
	return getPixVal(img, pnt.x, pnt.y);
}
int CUDAGlobalPmat::getPixVal(host_vector<int>& img, int x, int y)
{
	int indx = getIndex(x, y);
	return (int)img[indx];
}

int CUDAGlobalPmat::getPVal(Point p1, Point p2)
{
	
	return 0;
}

int CUDAGlobalPmat::getPFromImages(Point p1, Point p2)
{
	host_vector<int> p(m_patch*m_patch);// patch*patch vector of
	return 0;
}


