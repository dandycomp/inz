#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <iostream>

#include "brut_corr.hpp"
#include "gl_mat.hpp"




//#include "fast_cc.hpp"
#include "recursiveFastCC.h"

#if 0
#include <string>
#include <fstream>
#include <cstring>
#include "cuda_code.cuh"
#endif

//#include "cuda_thrust.cuh"
using namespace std;
using namespace cv;

//gpu::meanStdDev
int main()
{
	
	Mat img1 = imread("img/img1.jpg");
	Mat img2 = imread("img/img2.jpg");

	cvtColor(img1, img1, CV_RGB2GRAY);
	cvtColor(img2, img2, CV_RGB2GRAY);

	int patch = 7;
	int subImg = 15;
	
	//CUDAGlobalPmat cudaP;
	//cudaP.setParameters(img1, img2, subImg, patch);
	//cudaP.run();

	BrutCorr brCorr;
	brCorr.setParameters(img1, img2, subImg, patch);
	//brCorr.correlate();

    GlSumTbl gl1, gl2;
    gl1.setParameters(img1, subImg, patch);
    gl1.createGlSums();

    gl2.setParameters(img2, subImg, patch);
    gl2.createGlSums();


    FastCC fcc;
    fcc.setParameters(gl1, gl2);
	//fcc.runFastCC();
    //fcc.recursiveFastCC();
	//fcc.recursiveFastCCStructure();

	//cout << " Press any key" << endl;
	//cin.get();

	cout << "*****END*******" << endl;
	return 1;
}