#pragma once

//*****non cuda libraries *********
#include "gl_mat.hpp"

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
//*****cuda libraries *********
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel_functions.cuh"

#include <ctime>
#include "outpputDataStructure.h"

class CudaInterface{
public:
	CudaInterface();
	~CudaInterface();
	void setParameters(Mat&, Mat&);
	void run();
	void deviceInfo();
	void fastCudaCorrelation();
private:
	void cudaDrawDirectionHeatMap(vector<vector<OutpStr>>);
private:
	int * m_img1_dev;
	int * m_img2_dev;
};
