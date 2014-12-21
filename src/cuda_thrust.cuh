#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2\gpu\device\common.hpp"

#include "thrust/version.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "opencv2/gpu/gpu.hpp"


#include <iostream>
using namespace std;
using namespace cv;
using thrust::device_vector;
using thrust::host_vector;


class CUDAGlobalPmat{
public:
	CUDAGlobalPmat(){}
	~CUDAGlobalPmat();
	//CUDAGlobalPmat(CUDAGlobalPmat&);
	void info();
	void setParameters(Mat& img1, Mat& img2, int subImg, int patchSize);
	void run();

private:
	int getPFromImages(Point, Point);
	int getPixVal(host_vector<int>& img, Point pnt  );
	int getPixVal(host_vector<int>& img, int x, int y);
	int getPVal(Point, Point);
	host_vector< int>matToHostVector(Mat&);

	int getIndex(int, int);
	int getIndex(Point);
	int m_subImg;//size of sub image
	int m_patch;// size of patch

	Size m_sizeImg;


public:
	host_vector< int> m_imgVec1; // Kontener przechowuj¹cy obrazy
	host_vector< int> m_imgVec2;
	host_vector<host_vector<int>>m_PGlMat; // size is smaller than m_imgVec1 or m_imgVec2
};