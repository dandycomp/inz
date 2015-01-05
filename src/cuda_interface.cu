#include "cuda_interface.cuh"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_profiler_api.h>

using namespace cv;
using namespace std;

CudaInterface::CudaInterface(){
	m_img1_dev = NULL;
	m_img2_dev = NULL;

	m_glSum1_dev = NULL;
	m_glSum2_dev = NULL;

	m_glSqSum1_dev = NULL;
	m_glSqSum2_dev = NULL;
}

CudaInterface::~CudaInterface(){

	free(m_img1_host);
	free(m_img2_host);

	cudaFree(m_img1_dev);
	cudaFree(m_img2_dev);

	cudaFree(m_glSum1_dev);
	cudaFree(m_glSum2_dev);

	cudaFree(m_glSqSum1_dev);
	cudaFree(m_glSqSum2_dev);
}

void CudaInterface::setParameters(
	GlSumTbl& glimg1,
	GlSumTbl& glimg2){

	m_imgWidth = glimg1.m_img.cols;
	m_imgHeight = glimg1.m_img.rows;

	m_glWidth = glimg1.m_glSum.cols;
	m_glHeight = glimg1.m_glSum.rows;

	m_patch = glimg1.m_patch;
	m_subImg = glimg1.m_subImg;
	size_t numImageBytes = m_imgWidth* m_imgHeight * sizeof(int);
	size_t numGlSumBytes = m_glWidth *  m_glHeight * sizeof(int);

	m_img1_host = (int*)malloc(numImageBytes);//index format image
	m_img2_host = (int*)malloc(numImageBytes);

	m_glSum1_host = (int*)malloc(numGlSumBytes);
	int* m_glSum2_host = (int*)malloc(numGlSumBytes);

	m_glSqSum1_host = (int*)malloc(numGlSumBytes);
	int * m_glSqSum2_host = (int*)malloc(numGlSumBytes);

	//filling arrays of image data
	for (int i = 0; i < m_imgHeight; i++)
		for (int j = 0; j < m_imgWidth; j++)
		{
		int index = i*m_imgWidth + j;
		int tempPixelValue1 = glimg1.m_img.at<uchar>(i, j);
		int tempPixelValue2 = glimg2.m_img.at<uchar>(i, j);
		m_img1_host[index] = tempPixelValue1;
		m_img2_host[index] = tempPixelValue2;
		}

	//filling array of global sums and global squared sums
	for (int i = 0; i < m_glHeight; i++)
		for (int j = 0; j < m_glWidth; j++)
		{
		int index = i*m_glWidth + j;
		int tempGlSum1 = glimg1.m_glSum.at<float>(i, j);
		int tempGlSum2 = glimg2.m_glSum.at<float>(i, j);
		int tempGlSqSum1 = glimg1.m_glSqSum.at<float>(i, j);
		int tempGlSqSum2 = glimg2.m_glSqSum.at<float>(i, j);

		m_glSum1_host[index] = tempGlSum1;
		m_glSum2_host[index] = tempGlSum2;





		m_glSqSum1_host[index] = tempGlSqSum1;
		m_glSqSum2_host[index] = tempGlSqSum2;

		}


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		this->~CudaInterface();
	}

	cudaMalloc((void**)&m_img1_dev, numImageBytes);
	cudaMalloc((void**)&m_img2_dev, numImageBytes);

	cudaMalloc((void**)&m_glSum1_dev, numGlSumBytes);
	cudaMalloc((void**)&m_glSum2_dev, numGlSumBytes);

	cudaMalloc((void**)&m_glSqSum1_dev, numGlSumBytes);
	cudaMalloc((void**)&m_glSqSum2_dev, numGlSumBytes);


	cudaMemcpy(m_img1_dev, m_img1_host, numImageBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_img2_dev, m_img2_host, numImageBytes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_glSum1_dev, m_glSum1_host, numGlSumBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_glSum2_dev, m_glSum2_host, numGlSumBytes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_glSqSum1_dev, m_glSqSum1_host, numGlSumBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_glSqSum2_dev, m_glSqSum2_host, numGlSumBytes, cudaMemcpyHostToDevice);

	//free(m_glSum1_host);
	free(m_glSum2_host);
	//free(m_glSqSum1_host);
	free(m_glSqSum2_host);
}

void CudaInterface::array2Mat(float* arr){
	//zakladamy tylko poprawny rozmiar orazu
	int glWidth = IMAGE_WIDTH - PATCH_SIZE + 1;
	int glHeight = IMAGE_HEIGHT - PATCH_SIZE + 1;
	cv::Mat outpImg = cv::Mat::zeros(glHeight, glWidth, CV_8U);
	for (int i = 0; i < glHeight; i++)
		for (int j = 0; j < glWidth; j++)
		{
		int index = i*glWidth + j;
		float corr = arr[index];
		corr = 255 * (corr + 1.0) / 2.0;
		outpImg.at<uchar>(i, j) = (int)corr;
		}
	cv::imwrite("cudaOutp.jpg", outpImg);
}
void CudaInterface::run(){
	this->correlate();
	//getBestCorrFromArea_fast(20, 20);
}

void CudaInterface::correlate()
{
cudaError start = cudaProfilerStart();

int glWidth = IMAGE_WIDTH - PATCH_SIZE + 1;
int glHeight = IMAGE_HEIGHT - PATCH_SIZE + 1;
float* corrMat = new float[glWidth*glHeight];

int finishedPercent = 0;

for (int col = 0; col < glWidth; col++)
	for (int row = 0; row < glHeight; row++)
	{
	int index = row*glWidth + col;
	//corrMat[index] = getBestCorrFromArea(col + PATCH_SIZE / 2, row + PATCH_SIZE / 2);
	//corrMat[index] = simpleGetBestCorrFromArea(col + PATCH_SIZE / 2, row + PATCH_SIZE / 2);
	corrMat[index] = getBestCorrFromArea_fast(col + PATCH_SIZE / 2, row + PATCH_SIZE / 2);
	//cin.get();

	int actual = (col*glHeight + row) * 100
		/ (glWidth * glHeight);

	if (actual > finishedPercent)
	{
		finishedPercent = actual;
		cout << finishedPercent << " % " << endl;
	}
	}

cudaError stop = cudaProfilerStop();
array2Mat(corrMat);
}

float CudaInterface::getBestCorrFromArea_fast(int x, int y){

	// counting parameters of central patch of first image to simplfy and 
	// optimize algorithm. We need:
	//patch*patch array of pixel intensity values
	//medium value of this pixels
	//standart deviation of this patch

	int* im1_patch = new int[PATCH_SIZE*PATCH_SIZE];
	for (int row = 0; row < PATCH_SIZE; row++)
		for (int col = 0; col < PATCH_SIZE; col++)
		{
		int x_ = x - PATCH_SIZE / 2 + col;
		int y_ = y - PATCH_SIZE / 2 + row;

		int index = y_ * IMAGE_WIDTH + x_;
		int temp = m_img1_host[index];
		im1_patch[row * PATCH_SIZE + col] = temp;
		}

	int glImageWidth = IMAGE_WIDTH - PATCH_SIZE + 1;
	int index = x - PATCH_SIZE / 2 + (y - PATCH_SIZE / 2) * glImageWidth;
	int im1_patch_glSum = m_glSum1_host[index];
	int im1_patch_glSqSum = m_glSqSum1_host[index];

	dim3 threads(SUB_IMG, SUB_IMG);
	dim3 blocks(1, 1);

	int* im1_patch_device = NULL;
	cudaMalloc((void**)&im1_patch_device, PATCH_SIZE*PATCH_SIZE*sizeof(int));
	cudaMemcpy(im1_patch_device, im1_patch, PATCH_SIZE*PATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	int area_side = SUB_IMG - PATCH_SIZE + 1;
	float* dev_corr = NULL;
	cudaMalloc((void**)&dev_corr, area_side*area_side*sizeof(float));

	cudaGetBestCorrelate_fast << <blocks, threads >> >(m_img2_dev, m_glSum2_dev, m_glSqSum2_dev,
		im1_patch_device, x, y,
		im1_patch_glSum, im1_patch_glSqSum, dev_corr);

	float* host_corr =  new float [area_side*area_side];
	cudaMemcpy(host_corr, dev_corr, area_side*area_side*sizeof(float), cudaMemcpyDeviceToHost);

	float max_corr = -1;
	for(int i = 0; i < area_side*area_side; i++)
	{
		max_corr = host_corr[i] > max_corr? host_corr[i] : max_corr;
		//cout << " Corr = " << host_corr[i] << endl;
	}
	//cout << " Max correlation = " << max_corr << endl;
	return max_corr;
}


float CudaInterface::getBestCorrFromArea(int x, int y){

	int area_side = SUB_IMG - PATCH_SIZE + 1; // one side of results area matrix
	int area_size = area_side*area_side;// size of area

	dim3 threads(PATCH_SIZE, PATCH_SIZE);
	dim3 blocks(area_side, area_side);

	int *host_P_output = new int[area_size];// size of area around pixel
	for (int i = 0; i < area_size; i++)
		host_P_output[i] = 0.0;

	int *dev_P_output;
	cudaMalloc((void**)&dev_P_output, area_size*sizeof(int));
	cudaMemcpy(dev_P_output, host_P_output, area_size*sizeof(int), cudaMemcpyHostToDevice);

	cudaGetAll_P_FromArea << <blocks, threads >> >(m_img1_dev, m_img2_dev,
		x, y, dev_P_output);

	cudaDeviceSynchronize();

	//cudaMemcpy(host_P_output, dev_P_output, area_size*sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < area_size; i++)
	//	cout << " p = " << host_P_output[i] << endl;

	//cin.get();



	float *host_output = new float[area_size];// size of area around pixel
	for (int i = 0; i < area_size; i++)
		host_output[i] = 0.0;

	float *dev_outp;
	cudaMalloc((void**)&dev_outp, area_size*sizeof(float));
	cudaMemcpy(dev_outp, host_output, area_size*sizeof(float), cudaMemcpyHostToDevice);


	dim3 threads_2(area_side, area_side);
	cudaGetBestCorrelateFromArea << < 1, threads_2 >> >(dev_P_output, m_glSum1_dev,
		m_glSum2_dev, m_glSqSum1_dev, m_glSqSum2_dev,
		x, y, dev_outp);

	cudaDeviceSynchronize();
	cudaMemcpy(host_output, dev_outp, area_size*sizeof(float), cudaMemcpyDeviceToHost);

	float max_cc = -1;
	for (int i = 0; i < area_size; i++){
		//cout << " host_output[i] = " << host_output[i] << endl;
		max_cc = host_output[i] > max_cc ? host_output[i] : max_cc;
	}

	cudaFree(dev_P_output);
	cudaFree(dev_outp);
	return max_cc;
}

float CudaInterface::simpleGetBestCorrFromArea(int x, int y){
	
	// counting parameters of central patch of first image to simplfy and 
	// optimize algorithm. We need:
	//patch*patch array of pixel intensity values
	//medium value of this pixels
	//standart deviation of this patch

	int* im1_patch = new int[PATCH_SIZE*PATCH_SIZE];
	int sum = 0;
	for (int row = 0; row < PATCH_SIZE; row++)
		for (int col = 0; col < PATCH_SIZE; col++)
		{
		int x_ = x - PATCH_SIZE / 2 + col;
		int y_ = y - PATCH_SIZE / 2 + row;

		int index = y_ * IMAGE_WIDTH + x_;
		int temp = m_img1_host[index];
		im1_patch[row * PATCH_SIZE + col] = temp;
		sum += temp;
		}

	int medium1 = (float)sum / ((float)PATCH_SIZE*(float)PATCH_SIZE);
	float stDev1 = 0;
	for (int i = 0; i < PATCH_SIZE*PATCH_SIZE; i++)
		stDev1 += pow((float)im1_patch[i] - medium1,2);

	int area_side = SUB_IMG - PATCH_SIZE + 1; // one side of results area matrix
	dim3 threads_(SUB_IMG, SUB_IMG);

	dim3 blocks_(1, 1);
	int* im1_patch_device = NULL;
	cudaMalloc((void**)&im1_patch_device, PATCH_SIZE*PATCH_SIZE*sizeof(int));
	cudaMemcpy(im1_patch_device, im1_patch, PATCH_SIZE*PATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	float* dev_corr = NULL;
	cudaMalloc((void**)&dev_corr, area_side*area_side*sizeof(float));


	cudaSimpleGetBestCorrelate<<<blocks_,threads_>>>(m_img2_dev,x,y,
		im1_patch_device, medium1, stDev1, dev_corr);

	float* host_corr =  new float [area_side*area_side];
	cudaMemcpy(host_corr, dev_corr, area_side*area_side*sizeof(float), cudaMemcpyDeviceToHost);
	
	float max_corr = -1;
	for(int i = 0; i < area_side*area_side; i++)
	{
		max_corr = host_corr[i] > max_corr? host_corr[i] : max_corr;
		//cout << " Corr = " << host_corr[i] << endl;
	}

	//cout << " Max correlation = " << max_corr << endl;

	return max_corr;

#if 0





	dim3 threads(PATCH_SIZE, PATCH_SIZE);
	dim3 blocks(area_side, area_side);

	int *host_P_output = new int[area_size];// size of area around pixel
	for (int i = 0; i < area_size; i++)
		host_P_output[i] = 0.0;

	int *dev_P_output;
	cudaMalloc((void**)&dev_P_output, area_size*sizeof(int));
	cudaMemcpy(dev_P_output, host_P_output, area_size*sizeof(int), cudaMemcpyHostToDevice);

	cudaGetAll_P_FromArea << <blocks, threads >> >(m_img1_dev, m_img2_dev,
		x, y, dev_P_output);

	cudaDeviceSynchronize();

	//cudaMemcpy(host_P_output, dev_P_output, area_size*sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < area_size; i++)
	//	cout << " p = " << host_P_output[i] << endl;

	//cin.get();



	float *host_output = new float[area_size];// size of area around pixel
	for (int i = 0; i < area_size; i++)
		host_output[i] = 0.0;

	float *dev_outp;
	cudaMalloc((void**)&dev_outp, area_size*sizeof(float));
	cudaMemcpy(dev_outp, host_output, area_size*sizeof(float), cudaMemcpyHostToDevice);


	dim3 threads_2(area_side, area_side);
	cudaGetBestCorrelateFromArea << < 1, threads_2 >> >(dev_P_output, m_glSum1_dev,
		m_glSum2_dev, m_glSqSum1_dev, m_glSqSum2_dev,
		x, y, dev_outp);

	cudaDeviceSynchronize();
	cudaMemcpy(host_output, dev_outp, area_size*sizeof(float), cudaMemcpyDeviceToHost);

	float max_cc = -1;
	for (int i = 0; i < area_size; i++){
		//cout << " host_output[i] = " << host_output[i] << endl;
		max_cc = host_output[i] > max_cc ? host_output[i] : max_cc;
	}

	cudaFree(dev_P_output);
	cudaFree(dev_outp);
	return max_cc;
#endif
}

void CudaInterface::simpleCorrelate()
{
	cudaError start = cudaProfilerStart();
	int outpWidth = IMAGE_WIDTH - PATCH_SIZE + 1;
	int outpHeight = IMAGE_HEIGHT - PATCH_SIZE + 1;
	float* corrMat = new float[outpWidth*outpHeight];

	int finishedPercent = 0;

	for (int col = 0; col < outpWidth; col++)
		for (int row = 0; row < outpHeight; row++)
		{
		int index = row*outpWidth + col;
			
		//corrMat[index] = getBestCorrFromArea(col + PATCH_SIZE / 2, row + PATCH_SIZE / 2);

		int actual = (col*outpHeight + row) * 100
			/ (outpWidth * outpHeight);

		if (actual > finishedPercent)
		{
			finishedPercent = actual;
			cout << finishedPercent << " % " << endl;
		}
		}

	cudaError stop = cudaProfilerStop();
	array2Mat(corrMat);

}


void CudaInterface::deviceInfo(){
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

