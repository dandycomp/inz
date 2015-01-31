#ifndef OUTP_STRUCT
#define OUTP_STRUCT

#include "opencv2\core\core.hpp"
#include "outpputDataStructure.h"
using namespace std;
using namespace cv;
struct OutpStr{
	Point m_point1;
	Point m_point2;
	float m_CCcoeff;
	OutpStr(Point p1, Point p2, float coeff) : m_point1(p1), m_point2(p2), m_CCcoeff(coeff){}
	OutpStr() :m_point1(Point(0, 0)), m_point2(Point(0, 0)), m_CCcoeff(0){}
	void print(){ cout << " Output Structure: " << m_point1 << " " << m_point2 << " " << m_CCcoeff << endl; }
	/*OutpStr(OutpStr& a){
		this->m_point1 = a.m_point1;
		this->m_point2 = a.m_point2;
		this->m_CCcoeff = a.m_CCcoeff;
	}*/
};


#endif