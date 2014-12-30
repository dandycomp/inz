#include "cuda_interface.cuh"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

//#include <cuda_profiler_api.h>

using namespace cv;
using namespace std;

//#define PATCH_SIZE  7          // submatrix size
static const int PATCH_SIZE = 7;// rozwiazac ten problem za posrednictwem parametrow wejsciowych
static const int SUB_IMG = 15;
static const int IMAGE_WIDTH = 133;
static const int IMAGE_HEIGHT = 98;

__global__ void cudaGetBestCorrelateFromArea(int* p_values, int* glsum1,
	int* glsum2, int* glSqSum1, int* glSqSum2,
	int x, int y, float* outp)
{
	// póŸniej trzeba bêdzie obliczenie dla pierwszego patcha wynieœæ poza funkcjê i przekazywaæ jako argument
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;
	const int distortion = block_side / 2;// przesuniêcie pocz¹tku odliczania od punktu x,y

	float f_value = 0.0;
	float g_value = 0.0;
	float q_value = 0.0;

	const int gl_dist = PATCH_SIZE / 2;
	int y_ = ty - block_side / 2 - gl_dist;// przesuniecie punktu pocz¹tku odliczania wzglêdem x,y
	int x_ = tx - block_side / 2 - gl_dist;

	int gl_width = IMAGE_WIDTH - PATCH_SIZE + 1;// width of global images
	
	int index1 = (y - gl_dist) * gl_width + (x - gl_dist);
	int index2 = (y + y_)*gl_width + x + x_;

	float glSum1 = (float)glsum1[index1];
	float glSum2 = (float)glsum2[index2];
	float pixelInPatch = (float)PATCH_SIZE *(float)PATCH_SIZE;

	f_value = (float)glSqSum1[index1] - glSum1 * glSum1 / pixelInPatch;// moze byc problem z zaokragleniem
	g_value = (float)glSqSum2[index2] - glSum2 * glSum2 / pixelInPatch;// moze byc problem z zaokragleniem
	q_value = glSum1 * glSum2 / pixelInPatch;
	float p_value = p_values[ty*block_side + tx];

	outp[ty*block_side + tx] = (p_value - q_value) / sqrt(f_value * g_value);
}


__global__ void cudaGetAll_P_FromArea(int *img1, int* img2,
	int x, int y, int* outpPvector)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int block_side = SUB_IMG - PATCH_SIZE + 1;

	int dist_x = bx - block_side / 2;
	int dist_y = by - block_side / 2;

	int p_sum = 0;

	__shared__ int shared_img1[PATCH_SIZE*PATCH_SIZE];
	__shared__ int shared_img2[PATCH_SIZE*PATCH_SIZE];

	int thread_X = x + tx - PATCH_SIZE / 2;// uwzglêdniamy i przesuniêcie centralnego piksela oraz ca³y patch dooko³a tego piksela
	int thread_Y = y + ty - PATCH_SIZE / 2;

	shared_img1[tx + PATCH_SIZE*ty] = img1[thread_Y*IMAGE_WIDTH + thread_X];
	shared_img2[tx + PATCH_SIZE*ty] = img2[(thread_Y + dist_y)*IMAGE_WIDTH + (thread_X + dist_x)];

	__syncthreads();

	for (int k = 0; k < PATCH_SIZE*PATCH_SIZE; k++)
		p_sum += shared_img1[k] * shared_img2[k];

	// Synchronize to make sure that the preceding
	// computation is done before loading two new
	// sub-matrices of A and B in the next iteration
	__syncthreads();

	outpPvector[by*block_side + bx] = p_sum;
}


CudaInterface::CudaInterface(){
	m_img1_dev = NULL;
	m_img2_dev = NULL;

	m_glSum1_dev = NULL;
	m_glSum2_dev = NULL;

	m_glSqSum1_dev = NULL;
	m_glSqSum2_dev = NULL;
}

CudaInterface::~CudaInterface(){

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

	int *img1_host = (int*)malloc(numImageBytes);//index format image
	int *img2_host = (int*)malloc(numImageBytes);

	int * m_glSum1_host = (int*)malloc(numGlSumBytes);
	int * m_glSum2_host = (int*)malloc(numGlSumBytes);

	int * m_glSqSum1_host = (int*)malloc(numGlSumBytes);
	int * m_glSqSum2_host = (int*)malloc(numGlSumBytes);

	//filling arrays of image data
	for (int i = 0; i < m_imgHeight; i++)
		for (int j = 0; j < m_imgWidth; j++)
		{
		int index = i*m_imgWidth + j;
		int tempPixelValue1 = glimg1.m_img.at<uchar>(i, j);
		int tempPixelValue2 = glimg2.m_img.at<uchar>(i, j);
		img1_host[index] = tempPixelValue1;
		img2_host[index] = tempPixelValue2;
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


	cudaMemcpy(m_img1_dev, img1_host, numImageBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_img2_dev, img2_host, numImageBytes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_glSum1_dev, m_glSum1_host, numGlSumBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_glSum2_dev, m_glSum2_host, numGlSumBytes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_glSqSum1_dev, m_glSqSum1_host, numGlSumBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_glSqSum2_dev, m_glSqSum2_host, numGlSumBytes, cudaMemcpyHostToDevice);

	free(img1_host);
	free(img2_host);
	free(m_glSum1_host);
	free(m_glSum2_host);
	free(m_glSqSum1_host);
	free(m_glSqSum2_host);
}

void CudaInterface::array2Mat(float* arr){
	//zakladamy tylko poprawny rozmiar orazu
	int glWidth = IMAGE_WIDTH - PATCH_SIZE + 1;
	int glHeight = IMAGE_HEIGHT - PATCH_SIZE + 1;
	cv::Mat outpImg = cv::Mat::zeros(glHeight,glWidth, CV_8U);
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
#if 1
	//cudaError start = cudaProfilerStart();

	int glWidth  = IMAGE_WIDTH - PATCH_SIZE + 1;
	int glHeight = IMAGE_HEIGHT - PATCH_SIZE + 1;
	float* corrMat = new float[glWidth*glHeight];

	int finishedPercent = 0;

	for (int col = 0; col < glWidth; col++)
		for (int row = 0; row < glHeight; row++)
		{
		int index = row*glWidth + col;
		corrMat[index] = getBestCorrFromArea(col + PATCH_SIZE / 2, row + PATCH_SIZE / 2);
		int actual = (col*glHeight + row) * 100
			/ (glWidth * glHeight);

		if (actual > finishedPercent)
		{
			finishedPercent = actual;
			cout << finishedPercent << " % " << endl;
		}
		}

	//cudaError stop = cudaProfilerStop();
	array2Mat(corrMat);
#endif
	//cout << " best corr 50,50 = " << getBestCorrFromArea(Point(50, 50)) << endl;
	cout << " End " << endl;
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