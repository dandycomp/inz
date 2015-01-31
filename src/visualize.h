#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\contrib\contrib.hpp"
#include <iostream>
#include <vector>
#include "outpputDataStructure.h"
#include "parameters.h"

using namespace std;
using namespace cv;

class VisualizeCC{
public:
	VisualizeCC(){}
	void drawDistHeatMap(vector<vector<OutpStr>>, string = "img/wizualizeDist.jpg");
	void drawCorrelationHeatMap(vector<vector<OutpStr>>, string = "img/wizualizeCorr.jpg");
	void drawDirectionHeatMap(vector<vector<OutpStr>>, string = "img/wizualizeDirection.jpg");
};
