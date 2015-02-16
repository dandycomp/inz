#ifndef GL_MAT_PREPROCESSING_HPP_INCLUDED
#define GL_MAT_PREPROCESSING_HPP_INCLUDED

#include "opencv2\core\core.hpp"

#include <iostream>

using namespace std;
using namespace cv;


class GlSumTbl{

public:
	GlSumTbl(){ setZero();}
	GlSumTbl(GlSumTbl&);
    bool data();
    void setParameters(Mat&, int subImg, int patch);
    void createGlSums();

private:

    void createGlobalSum(Mat& sumInt, Mat& sqSumInt);
    int getSumFromIntegral(Mat&, Point);
    void toIntegral(Mat& outpSum, Mat& outpSqSum);
    void setZero();

public:
    Mat m_img;
    Mat m_glSum;
    Mat m_glSqSum;
    int m_patch;
    int m_subImg;

};

#endif // GL_MAT_PREPROCESSING_HPP_INCLUDED
