#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <iostream>

#include "brut_corr.hpp"
#include "gl_mat.hpp"
#include "parameters.h"

//#include "fast_cc.hpp"
#include "recursiveFastCC.h"
#include "cuda_interface.cuh"

using namespace std;
using namespace cv;

time_t ncc_start, ncc_stop, fastRec_start, fastRec_stop, cuda_start, cuda_stop;
double NCCtime, FastRecTime, cudaTime;

#if 1
int main(int argc, char *argv[])
{
	// Ladowanie obrazow
	Mat img1 = imread(image1name);
	Mat img2 = imread(image2name);

	// interpolacja
	resize(img1, img1, Size(), RESIZE_COEFF, RESIZE_COEFF, INTER_CUBIC);
	resize(img2, img2, Size(), RESIZE_COEFF, RESIZE_COEFF, INTER_CUBIC);

	if (!img1.data)
	{
		img1 = imread("../../" + image1name);
		img2 = imread("../../" + image2name);
	}

	// zmiana na postac szarosciowa
	cvtColor(img1, img1, CV_RGB2GRAY);
	cvtColor(img2, img2, CV_RGB2GRAY);


	ncc_start = clock();

	//inicjalizacja NCC
	NCC ncCorr;
	ncCorr.setParameters(img1, img2, SUB_IMG, PATCH_SIZE);
	//funkcja wykonujaca obliczenia cross korelacji metoda NCC
    ncCorr.correlate();

	ncc_stop = clock();
	NCCtime = ncc_stop - ncc_start;

	fastRec_start = clock();

	//tworzenie sum globalnych
    GlSumTbl gl1, gl2;
	gl1.setParameters(img1, SUB_IMG, PATCH_SIZE);
    gl1.createGlSums();

	gl2.setParameters(img2, SUB_IMG, PATCH_SIZE);
    gl2.createGlSums();

	//inicjalizacja metody przyspieszenia sum globalnych z przyspieszeniem rekursywnym
	FastCC fcc;
	fcc.setParameters(gl1, gl2);
	//funkcja obliczenia cross korelacji metoda FastRecursive
	//fcc.recursiveFastCC();

	fastRec_stop = clock();
	FastRecTime = fastRec_stop - fastRec_start;

	cuda_start = clock();

	// inicjalizacja parametrow do obliczenia CUDAFastCorrelation
	CudaInterface cuda_interf;
	cuda_interf.setParameters(img1, img2);
	//cuda_interf.deviceInfo();
	// funkcja obliczajaca cross korelacje metoda CUDAFastCorrelation
	//cuda_interf.fastCudaCorrelation();

	cuda_stop = clock();
	cudaTime = cuda_stop - cuda_start;

	cout << " Rozmiar obrazu: " << img1.size() << " rozmiar patcha: " << PATCH_SIZE << " rozmiar ROI: " << SUB_IMG << endl;
	cout << "Czas wykonania w cuda = " << cudaTime << " czas na FastRecursive " << FastRecTime << endl;
	return 0;
}
#endif