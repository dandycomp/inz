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

time_t fastRec_start, fastRec_stop, cuda_start, cuda_stop;
double NCCtime, FastRecTime, cudaTime;

//gpu::meanStdDev
int main(int argc, char *argv[])
{
	Mat img1 = imread(image1name);
	Mat img2 = imread(image2name);
#if 0


	Rect myROI1(0, 0, 1500, 1500);
	cv::Mat cropped1 = img1(myROI1);
	cv::Mat cropped2 = img2(myROI1);
	imwrite("img/test/ref1.tif", cropped1);
	imwrite("img/test/smudge1.tif", cropped2);
	myROI1 = Rect(800, 800, 1500, 1500);
	cropped1 = img1(myROI1);
	cropped2 = img2(myROI1);
	imwrite("img/test/ref2.tif", cropped1);
	imwrite("img/test/smudge2.tif", cropped2);
	return 0;


	img / dziura1.tif"
	Rect myROI1(84, 460, 225, 164);

	cv::Mat cropped1 = img1(myROI1);
	cv::Mat cropped2 = img1(myROI1);
	imwrite("img/dziura1.tif", cropped1);
	imwrite("img/dziura2.tif", cropped2);
	return 0;

	return 0;


	// Setup a rectangle to define your region of interest
	Rect myROI(0, 0, 50, 50);
	cv::Mat cropped1 = img1(myROI);
	cv::Mat cropped2 = img2(myROI);
	imwrite("img/50_1.tif", cropped1);
	imwrite("img/50_2.tif", cropped2);

	myROI = Rect(0, 0, 100, 100);
	cropped1 = img1(myROI);
	cropped2 = img2(myROI);
	imwrite("img/100_1.tif", cropped1);
	imwrite("img/100_2.tif", cropped2);

	myROI = Rect(0, 0, 250, 250);
	cropped1 = img1(myROI);
	cropped2 = img2(myROI);
	imwrite("img/250_1.tif", cropped1);
	imwrite("img/250_2.tif", cropped2);

	myROI = Rect(0, 0, 500, 500);
	cropped1 = img1(myROI);
	cropped2 = img2(myROI);
	imwrite("img/500_1.tif", cropped1);
	imwrite("img/500_2.tif", cropped2);

	myROI = Rect(0, 0, 1000, 1000);
	cropped1 = img1(myROI);
	cropped2 = img2(myROI);
	imwrite("img/1000_1.tif", cropped1);
	imwrite("img/1000_2.tif", cropped2);


	myROI = Rect(0, 0, 2500, 2500);
	cropped1 = img1(myROI);
	cropped2 = img2(myROI);
	imwrite("img/2500_1.tif", cropped1);
	imwrite("img/2500_2.tif", cropped2);



	//Rect myROI_x(2, 0, 1000, 1000);
	//cv::Mat croppedImage1 = img1(myROI_x);

	//Rect myROI_y(0, 2, 1000, 1000);
	//cv::Mat croppedImage2 = img1(myROI_y);

	
	//imwrite("img/disp_x.tif", croppedImage1);
	//imwrite("img/disp_y.tif", croppedImage2);
	return 1;
#endif
	
#if 0
	Mat img3 = imread(image3name);
	Mat img4 = imread(image4name);
	Mat img5 = imread(image5name);
	Mat img6 = imread(image6name);
	Mat img7 = imread(image7name);
	Mat img8 = imread(image8name);
	Mat img9 = imread(image9name);
	Mat img10 = imread(image10name);

#if 0
	const std::string image1name = "img/HighContrastImages/HCY00 X03.tif";
	const std::string image2name = "img/HighContrastImages/HCY01 X00.tif";
	const std::string image3name = "img/HighContrastImages/HCY03 X00.tif";

	const std::string image4name = "img/HighContrastImages/HCY03 X03.tif";
	const std::string image5name = "img/HighContrastImages/HCY05 X05.tif";
	
	const std::string image6name = "img/HighContrastImages/HCY09 X00.tif";
	const std::string image7name = "img/HighContrastImages/HCY00 X09.tif";
	
	const std::string image8name = "img/HighContrastImages/HCY00 X05.tif";
	const std::string image9name = "img/HighContrastImages/HCY05 X01.tif";
	
	const std::string image10name = "img/HighContrastImages/HCY09 X09.tif";

#endif

	// Setup a rectangle to define your region of interest
	Rect myROI(0, 0, 100, 100);

	// Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
	cv::Mat croppedImage1 = img1(myROI);
	cv::Mat croppedImage2 = img2(myROI);
	cv::Mat croppedImage3 = img3(myROI);
	cv::Mat croppedImage4 = img4(myROI);
	cv::Mat croppedImage5 = img5(myROI);
	cv::Mat croppedImage6 = img6(myROI);
	cv::Mat croppedImage7 = img7(myROI);
	cv::Mat croppedImage8 = img8(myROI);
	cv::Mat croppedImage9 = img9(myROI);
	cv::Mat croppedImage10 = img10(myROI);

	imwrite("img/cuted/X3Y0_100.tif", croppedImage1);
	imwrite("img/cuted/X0Y1_100.tif", croppedImage2);
	imwrite("img/cuted/X0Y3_100.tif", croppedImage3);
	imwrite("img/cuted/X3Y3_100.tif", croppedImage4);
	imwrite("img/cuted/X5Y5_100.tif", croppedImage5);
	imwrite("img/cuted/X0Y9_100.tif", croppedImage6);
	imwrite("img/cuted/X9Y0_100.tif", croppedImage7);
	imwrite("img/cuted/X5Y0_100.tif", croppedImage8);
	imwrite("img/cuted/X1Y5_100.tif", croppedImage9);
	imwrite("img/cuted/X9Y9_100.tif", croppedImage10);
	return 1;
#endif

	resize(img1, img1, Size(), RESIZE_COEFF, RESIZE_COEFF, INTER_CUBIC);
	resize(img2, img2, Size(), RESIZE_COEFF, RESIZE_COEFF, INTER_CUBIC);


	if (!img1.data)
	{
		img1 = imread("../../" + image1name);
		img2 = imread("../../" + image2name);
	}

	cvtColor(img1, img1, CV_RGB2GRAY);
	cvtColor(img2, img2, CV_RGB2GRAY);


	NCC ncCorr;
	ncCorr.setParameters(img1, img2, SUB_IMG, PATCH_SIZE);
	//ncCorr.correlate();


	fastRec_start = clock();
    GlSumTbl gl1, gl2;
	gl1.setParameters(img1, SUB_IMG, PATCH_SIZE);
    gl1.createGlSums();

	gl2.setParameters(img2, SUB_IMG, PATCH_SIZE);
    gl2.createGlSums();

	FastCC fcc;
	fcc.setParameters(gl1, gl2);
	fcc.recursiveFastCCStructure();
	fastRec_stop = clock();
	FastRecTime = fastRec_stop - fastRec_start;

	cout << " Rozmiar obrazu: " << img1.size() << " rozmiar patcha: " << PATCH_SIZE << " rozmiar ROI: " << SUB_IMG << endl;

	cout << " czas na FastRecursive " << FastRecTime << endl;// " w cuda = " << cudaTime << endl;

	cuda_start = clock();
	CudaInterface cuda_interf;
	cuda_interf.setParameters(img1, img2);
	//cuda_interf.deviceInfo();
	cuda_interf.fastCudaCorrelation();

	cuda_stop = clock();
	cudaTime = cuda_stop - cuda_start;

	cout << " Rozmiar obrazu: " << img1.size() << " rozmiar patcha: " << PATCH_SIZE << " rozmiar ROI: " << SUB_IMG << endl;

	cout << "Czas wykonania w cuda = " << cudaTime << " czas na FastRecursive " << FastRecTime << endl;// " w cuda = " << cudaTime << endl;
	//cout << "Czas w NCC = " << NCCtime << " w FastRecursive = " << FastRecTime << " w cuda = " << cudaTime << endl;
	return 0;
}