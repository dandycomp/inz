#pragma once

//*****non cuda libraries *********
#include "gl_mat.hpp"

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
//#include "opencv2\imgproc\imgproc.hpp"
//#include "opencv2\highgui\highgui.hpp"
//*****cuda libraries *********
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel_functions.cuh"


#include <ctime>
#include <cuda_profiler_api.h>

#include "outpputDataStructure.h"

class CudaInterface{
public:
	CudaInterface();
	~CudaInterface();// dopisaæ usuniêcie macierzy
	//void setParameters(GlSumTbl& glimg1, GlSumTbl& glimg2);
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
