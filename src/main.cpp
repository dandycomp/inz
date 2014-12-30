#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <iostream>

#include "brut_corr.hpp"
#include "gl_mat.hpp"

//#include "fast_cc.hpp"
#include "recursiveFastCC.h"
#include "cuda_code.cuh"
#include "cuda_interface.cuh"

//#include "cuda_thrust.cuh"
using namespace std;
using namespace cv;

//gpu::meanStdDev
int main()
{
	
	Mat img1 = imread("img/col.jpg");
	Mat img2 = imread("img/col1.jpg");
	//Mat img1 = imread("img/tiff/trxy_s2_00.tif");
	//Mat img2 = imread("img/tiff/trxy_s2_01.tif");

#if 0
	if (!img1.data)
	{
		img1 = imread("../../img/col.jpg");
		img2 = imread("../../img/col1.jpg");
	}
#endif

	cvtColor(img1, img1, CV_RGB2GRAY);
	cvtColor(img2, img2, CV_RGB2GRAY);



	int patch = 7;
	int subImg = 15;
	
	CUDAGlobalPmat cudaP;
	cudaP.setParameters(img1, img2, subImg, patch);
	cudaP.run();

	BrutCorr brCorr;
	brCorr.setParameters(img1, img2, subImg, patch);
	//brCorr.correlate();

    GlSumTbl gl1, gl2;
    gl1.setParameters(img1, subImg, patch);
    gl1.createGlSums();

    gl2.setParameters(img2, subImg, patch);
    gl2.createGlSums();


	CudaInterface cuda_interf;
	//cuda_interf.setParameters(gl1, gl2);
	cuda_interf.setParameters(gl1, gl2);
	cuda_interf.run();


    FastCC fcc;
    fcc.setParameters(gl1, gl2);
	//fcc.runFastCC();
    //fcc.recursiveFastCC();
	//fcc.recursiveFastCCStructure();

	//cout << " Press any key" << endl;
	//cin.get();

	cout << "*****END*******" << endl;
	return 0;
}