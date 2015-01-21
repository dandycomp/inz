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
	void fastCudaCorrelation();

private:

	int m_imgWidth;//nie potrzebne
	int m_imgHeight;//nie potrzebne

	int m_glWidth;//nie potrzebne
	int m_glHeight;//nie potrzebne

	int m_patch;//nie potrzebne
	int m_subImg;//nie potrzebne
private:
	cudaError setCudaParameters();
	float getBestCorrFromArea(Point p){ return getBestCorrFromArea(p.x, p.y); }
	float getBestCorrFromArea(int, int);
	float simpleGetBestCorrFromArea(int, int);
	float simpleGetBestCorrFromArea_v2(int, int);
	float simpleGetBestCorrFromArea_v3(int x, int y);
	float getBestCorrFromArea_fast(int, int);
	void array2Mat(float *);
	void array2Mat_v2(float *);
	void simpleCorrelate();
	void nccCorrelation(){ correlate(false); }
	void fastRecursiveCorrelation(){ correlate(true); }
	void correlate(bool);
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