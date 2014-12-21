#ifndef PMAT_HPP_INCLUDED
#define PMAT_HPP_INCLUDED

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

// nie trzymam zadnych referencji do obrazow. domyslnie pracuje na tym, ze pierwszy obraz jest sprawdzany wzgledem drugiego i
//nie rozwazam innych mozliwosci
class Pmat{
public:
    Pmat(){}
    Pmat(Pmat&);
    void setParameters(Point, int subImg, int patch);
    void setPMat(Mat& img1, Mat& img2);

    Pmat next_x_Pmat(Mat& img1, Mat& img2);//create Next nextPmatX from Pmat
    Pmat next_y_Pmat(Mat& img1, Mat& img2);//create Next nextPmatY from Pmat
    void writePmat();
    int getPFromMat(int i, int j);
    int  next_P_x(int col, int row, Mat&, Mat&);//create nextPx from P
    int  next_P_y(int col, int row, Mat&, Mat&);//create nextPy from P
private:
    int  getPFromImage(Mat&, Mat&, Point p1, Point p2);

public://docelowo private
    Mat m_Pmat;
    int m_subImg;//size of sub image
    int m_patch;// size of patch
    Point m_pnt;
};
#endif