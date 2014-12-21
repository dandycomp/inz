#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <cstring>
#include <string>
using namespace std;
using namespace cv;

namespace utils{
//***************EXPORT/IMPORT to file************
template <typename T>
void exportToFile(T &file, string name)
{
	ofstream outpfile(name.c_str(), ios::out | ios::binary);
	outpfile.write((char *) &file, sizeof(T));
	outpfile.close();
}

template < typename T >
T importFromFile(string name)
{
	ifstream inpFile(name.c_str(), ios::in | ios::binary);
	if (!inpFile.is_open())
		{
			cout << "Can'not open file" << endl;
		}

	T outp;
	inpFile.read((char *) &outp, sizeof(T));
	return outp;
}

template <typename T>
void exportMat(Mat img, string name)
{
	ofstream myfile (name.c_str());
	for(int i = 0; i < img.cols; i++)
	{
		myfile << "\n" << i  <<". ";
		for(int j = 0; j < img.rows; j++)
		{
			float temp = (float)img.at<T>(Point(j,i));
			myfile << " " << temp;
		}
	}
	myfile.close();
}
}

#endif // UTILS_HPP_INCLUDED
