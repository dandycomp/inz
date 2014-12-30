#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2\gpu\device\common.hpp"
#include "thrust\version.h"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
using namespace std;
using namespace cv;

class CUDAGlobalPmat{
public:
	CUDAGlobalPmat();
	~CUDAGlobalPmat();
    CUDAGlobalPmat(CUDAGlobalPmat&);
	void setParameters(Mat& img1, Mat& img2, int subImg, int patch);
	void run();
	void deviceInfo();

private:
	int  getPFromImages(uchar* img1, uchar* img2,
		int x1, int y1, int x2, int y2, int patch,
		int width);
	Mat getPmat(Point, Point);
public:
	//Mat* m_img1;
	//Mat* m_img2;
	uchar* m_arrayImg1;
	uchar* m_arrayImg2;
	uchar* m_cudaImg1;
	uchar* m_cudaImg2;

	Mat m_CUDAGlobalPmat; // PMAT[(N*N of P)] in GPU memory
	int m_cols;
	int m_subImg;//size of sub image
    int m_patch;// size of patch
};