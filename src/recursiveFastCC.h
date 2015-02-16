#ifndef RECURSIVE_FAST_CC_HPP_INCLUDED
#define RECURSIVE_FAST_CC_HPP_INCLUDED

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <ctime>

#include "gl_mat.hpp"
#include "p_mat.hpp"
#include "outpputDataStructure.h"
#include "visualize.h"

using namespace std;
using namespace cv;

class FastCC{

public:
	FastCC(){}
	void setParameters(GlSumTbl&, GlSumTbl&);
	void recursiveFastCC();
public:
	Mat m_corrMat;
private:
	void  create_F_and_G_mat();
	float getRecursiveCorrelate(Point, Point, float pVal);
	OutpStr bestCorrFromArea(Point, Pmat pmat);
	int   getP(Point, Point);
	float getQ(Point, Point);
	float getF(Point pnt);
	float getG(Point pnt);

private:
	Mat m_Fmat;
	Mat m_Gmat;
	GlSumTbl m_gl1;//float types
	GlSumTbl m_gl2;//float types
	int m_subImg;
	int m_patch;
	Pmat m_pVal;
	Pmat m_nextXpVal;
	Pmat m_nextYpVal;
};

#endif // FAST_CC_HPP_INCLUDED
