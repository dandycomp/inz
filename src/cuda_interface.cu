#include "cuda_interface.cuh"

using namespace cv;
using namespace std;


CudaInterface::CudaInterface(){
	m_img1_dev = NULL;
	m_img2_dev = NULL;
}

CudaInterface::~CudaInterface(){
	cudaFree(m_img1_dev);
	cudaFree(m_img2_dev);
}

void CudaInterface::setParameters(
	Mat& img1, Mat& img2){

	int img_width = img1.cols;
	int img_height = img1.rows;

	int* img1_host = new int[img_height* img_width];
	int* img2_host = new int[img_height* img_width];

	//filling arrays of image data
	for (int i = 0; i < img_height; i++)
		for (int j = 0; j < img_width; j++)
		{
		int index = i*img_width + j;
		int tempPixelValue1 = img1.at<uchar>(i, j);
		int tempPixelValue2 = img2.at<uchar>(i, j);
		img1_host[index] = tempPixelValue1;
		img2_host[index] = tempPixelValue2;
		}

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		this->~CudaInterface();
	}

	size_t numImageBytes = img_height* img_width * sizeof(int);
	//memory allocation on device
	cudaMalloc((void**)&m_img1_dev, numImageBytes);
	cudaMalloc((void**)&m_img2_dev, numImageBytes);


	//copying data to device memory
	cudaMemcpy(m_img1_dev, img1_host, numImageBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_img2_dev, img2_host, numImageBytes, cudaMemcpyHostToDevice);

	delete[] img1_host;
	delete[] img2_host;
}

void CudaInterface::fastCudaCorrelation(){
	time_t start, stop;
	double czas;
	start = clock();
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	int glWidth = IMAGE_WIDTH - SUB_IMG + 1;
	int glHeight = IMAGE_HEIGHT - SUB_IMG + 1;
	const int block_side = SUB_IMG - PATCH_SIZE + 1;

	float* corrMat_dev;
	//memory allocation on device
	cudaStatus = cudaMalloc((void**)&corrMat_dev, glWidth*glHeight*block_side*block_side*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of corrMat_dev failed! Can't allocate memory \n ");
		return;
	}

	dim3 blocks(IMAGE_WIDTH / SUB_IMG + 1, IMAGE_HEIGHT / SUB_IMG + 1);
	dim3 threads(SUB_IMG, SUB_IMG);

	int nr_of_iter = SUB_IMG * SUB_IMG;
	int finishedPercent = 0;

	int dist = SUB_IMG / 2 + 1; // starting distortion of first reference image patch

	cout << " \nWyznaczenie cross korelacji metoda CUDAFastCC, ukonczono: " << endl;
	for (int row = 0; row < SUB_IMG; row++)
		for (int col = 0; col < SUB_IMG; col++){

		int actual = (row*SUB_IMG + col) * 100
			/ (SUB_IMG*SUB_IMG);

		if (actual > finishedPercent)
		{
			finishedPercent = actual;
			cout << finishedPercent << " % " << endl;
		}
		
		cudaFastCorrelation << <blocks, threads >> >(m_img1_dev, m_img2_dev, col+dist, row + dist, corrMat_dev);
		cudaDeviceSynchronize();
		
		}

#if 1
	float* final_corrMat_dev;
	cudaStatus = cudaMalloc((void**)&final_corrMat_dev, glWidth*glHeight*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of final_corrMat_dev failed! Can't allocate memory \n ");
		return;
	}
	

	int* posMat_dev;
	cudaStatus = cudaMalloc((void**)&posMat_dev, glWidth*glHeight*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of posMat_dev failed! Can't allocate memory \n ");
		return;
	}
	
	int grid_size = 32;
	dim3 blocksToGetMax(glWidth/ grid_size + 1, glHeight/ grid_size + 1);
	dim3 threadsToGetMax(grid_size, grid_size);

	cudaGetMaxValues << <blocksToGetMax, threadsToGetMax >> >(corrMat_dev, final_corrMat_dev, posMat_dev);
	cudaDeviceSynchronize();


	stop = clock();
	float* final_corrMat_host = new float[glWidth*glHeight];
	cudaStatus = cudaMemcpy(final_corrMat_host, final_corrMat_dev,
		glWidth*glHeight*sizeof(float),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Can't copy final_corrMat_dev from device memory to host\n ");\
		cout << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	

	int* posMat_host = new int [glWidth*glHeight];
	cudaStatus = cudaMemcpy(posMat_host, posMat_dev,
		glWidth*glHeight*sizeof(int),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Can't copy posMat from device memory to host\n ");
		cout << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	
	vector<vector< OutpStr>> outpVector;
	outpVector.clear();
	for(int i = 0; i < glHeight; i++)
	{
		vector< OutpStr> colResults;
		colResults.clear();
		for (int j = 0; j < glWidth; j++)
		{
			int startPosition = j + i*glWidth;

			Point ref = Point(j, i);
			int position = posMat_host[startPosition];
			int x_dist = position - ((int)(position/block_side))*block_side - block_side / 2;
			int y_dist = position/block_side - block_side / 2;

			Point deform = Point(ref.x + x_dist, ref.y + y_dist);
			OutpStr tempStructure(ref, deform, final_corrMat_host[startPosition]);
			colResults.push_back(tempStructure);
		}
		outpVector.push_back(colResults);
	}
#endif

#if 0
	//start = clock();
	float* corrMat_host = new float[glWidth*glHeight*block_side*block_side];
	cudaStatus = cudaMemcpy(corrMat_host, corrMat_dev,
		block_side*block_side*glWidth*glHeight*sizeof(float),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Can't copy final_corrMat_dev from device memory to host\n ");
		cout << cudaGetErrorString(cudaStatus) << endl;
		this->~CudaInterface();
	}

	vector<vector< OutpStr>> outpVector;
	for(int i = 0; i < glHeight; i++)
	{
		vector< OutpStr> colResults;
		for (int j = 0; j < glWidth; j++)
		{
			float bestCorr = -1;
			Point pRecursive = Point(j, i);
			Point pDeformable = Point(0, 0);
			int startPosition = (j + i*glWidth)*block_side*block_side;

			int index_of_medium = startPosition + block_side / 2 + (block_side / 2 )*block_side;
			if (corrMat_host[index_of_medium] > 0.99)
			{
				OutpStr tempStructure(pRecursive, pRecursive, bestCorr);
				colResults.push_back(tempStructure);
			}
			else
			{
			for (int k = 0; k < block_side; k++)
			{
				for (int r = 0; r < block_side; r++)
				{
					int index = r + k * block_side + startPosition;
					float correlacja = corrMat_host[index];
					
					if (correlacja > bestCorr)
					{
						bestCorr = correlacja;
						pDeformable = Point(pRecursive.x + r - block_side / 2,
											pRecursive.y + k - block_side / 2);
					}
				}
			}
			
			OutpStr tempStructure(pRecursive, pDeformable, bestCorr);
			colResults.push_back(tempStructure);
			}
		}
		outpVector.push_back(colResults);
	}
	delete[] corrMat_host;
#endif

	
	czas = (stop - start);// / (double)1000;
	cout << "Czas obliczen w cuda nie uwzgledniajac kopiowan = " << czas << "ms. " << endl;

	cudaDrawDirectionHeatMap(outpVector);

}	

void CudaInterface::cudaDrawDirectionHeatMap(vector<vector<OutpStr>>data)
{
	int height = data.size();
	int width = data[0].size();
	Mat hsv(height, width, CV_8UC3);
	cvtColor(hsv, hsv, CV_RGB2HSV);

	//Mat correlate(height, width, CV_8U);
	//Mat distance(height, width, CV_8U);

	//cvtColor(hsv, hsv, CV_RGB2HSV);

	int max_dist = (SUB_IMG - PATCH_SIZE) / 2;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{

		OutpStr temp = data[i][j];
	
		int dx = temp.m_point2.x - temp.m_point1.x;
		int dy = temp.m_point2.y - temp.m_point1.y;
		//cout << dx << "  " << dy << endl << endl;


		float dist = sqrtf(pow(dx, 2) + pow(dy, 2));
		float corr = (1 + temp.m_CCcoeff) / 2;

		//uzywamy -dy zeby os 0,0 zaczynala od czytelnika strony
		float angle = atan2(-dy, dx) * 180 / 3.14;
		//cout << angle << endl << endl;
		if (angle < 0)
			angle = 360 + angle;
	
		//distance.at<uchar>(i, j) = 255-(int)(dist * 255.0 / (float)max_dist);
		//correlate.at<uchar>(i, j) = (int)(corr * 255);

		Vec3b tempHSV;
		tempHSV.val[0] = angle / 2;// (255 * (angle)) / 360;
		tempHSV.val[1] = (int)(dist * 255 / max_dist);
		tempHSV.val[2] = 255;// (int)(corr * 255);

		hsv.at<Vec3b>(i, j) = tempHSV;
		}
	cvtColor(hsv, hsv, CV_HSV2BGR);
	imwrite("img/CudaVisualizeDirection.jpg", hsv);

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

