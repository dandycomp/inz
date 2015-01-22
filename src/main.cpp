#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <iostream>

#include "brut_corr.hpp"
#include "gl_mat.hpp"
#include "parameters.h"

//#include "fast_cc.hpp"
#include "recursiveFastCC.h"
#include "cuda_code.cuh"
#include "cuda_interface.cuh"

using namespace std;
using namespace cv;


//gpu::meanStdDev
int main()
{
	Mat img1 = imread(image1name);
	Mat img2 = imread(image2name);

	resize(img1, img1, Size(), RESIZE_COEFF, RESIZE_COEFF, INTER_CUBIC);
	resize(img2, img2, Size(), RESIZE_COEFF, RESIZE_COEFF, INTER_CUBIC);

	//imwrite("img/outp.tif", img1);

	if (!img1.data)
	{
		img1 = imread("../../" + image1name);
		img2 = imread("../../" + image2name);
	}

	cvtColor(img1, img1, CV_RGB2GRAY);
	cvtColor(img2, img2, CV_RGB2GRAY);

	//CUDAGlobalPmat cudaP;
	//cudaP.setParameters(img1, img2, SUB_IMG, PATCH_SIZE);
	//cudaP.run();

	//NCC ncCorr;
	//ncCorr.setParameters(img1, img2, subImg, patch);
	//ncCorr.correlate();

    GlSumTbl gl1, gl2;
	gl1.setParameters(img1, SUB_IMG, PATCH_SIZE);
    gl1.createGlSums();

	gl2.setParameters(img2, SUB_IMG, PATCH_SIZE);
    gl2.createGlSums();

	CudaInterface cuda_interf;
	cuda_interf.setParameters(gl1, gl2);
	//cuda_interf.run();
	cuda_interf.fastCudaCorrelation();
	//cuda_interf.deviceInfo();

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