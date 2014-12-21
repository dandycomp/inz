#ifndef GL_SUM_HPP_INCLUDED
#define GL_SUM_HPP_INCLUDED

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;
class GlSum{

public:
    //GlSum(){ setZero();}

    void setParameters(Mat&, int subImg, int patch);
/*
private:
    void correlate();

    void getIntegral();
    void createGlobalSum();
*/
void setZero();
    private:
    Mat m_img;
    Mat m_glSum;
    Mat m_glSqSum;
    int m_patch;
    int m_subImg;
};


#endif // GL_SUM_HPP_INCLUDED
