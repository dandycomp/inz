#ifndef cuda_interface
#define cuda_interface

//*****non cuda libraries *********
#include "gl_mat.hpp"

#include "opencv2\core\core.hpp"
//#include "opencv2\imgproc\imgproc.hpp"
//#include "opencv2\highgui\highgui.hpp"
//*****cuda libraries *********
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel_functions.cuh"




class CudaInterface{
public:
	CudaInterface();
	~CudaInterface();// dopisaæ usuniêcie macierzy
	//void setParameters(GlSumTbl& glimg1, GlSumTbl& glimg2);
	void setParameters(GlSumTbl& glimg1, GlSumTbl& glimg2);
	void run();
	void deviceInfo();
private:

	int m_imgWidth;
	int m_imgHeight;

	int m_glWidth;
	int m_glHeight;

	int m_patch;
	int m_subImg;
private:
	cudaError setCudaParameters();
	float getBestCorrFromArea(Point p){ return getBestCorrFromArea(p.x, p.y); }
	float getBestCorrFromArea(int, int);
	float simpleGetBestCorrFromArea(int, int);
	float getBestCorrFromArea_fast(int, int);
	void array2Mat(float *);
	void simpleCorrelate();
	void correlate();
private:

	int* m_img1_host;
	int* m_img2_host;

	int* m_glSum1_host;
	int* m_glSqSum1_host;

	int * m_img1_dev;
	int * m_img2_dev;

	int * m_glSum1_dev;
	int * m_glSum2_dev;

	int * m_glSqSum1_dev;
	int * m_glSqSum2_dev;

};
#endif