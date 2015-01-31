#include "visualize.h"


void VisualizeCC::drawDistHeatMap(vector<vector<OutpStr>>data, string img_name)
{
	int height = data.size();
	int width = data[0].size();
	Mat outpMat = Mat::zeros(height, width, CV_8U);
	float maxVal = 0;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
		OutpStr temp = data[i][j];
		float dist = sqrtf(pow(temp.m_point1.x - 
			temp.m_point2.x, 2) + pow(
			temp.m_point1.y - temp.m_point2.y, 2));
		//cout << temp.m_point1.x << " " << temp.m_point2.x << " " <<
		//	temp.m_point1.y << " " << temp.m_point2.y << " " << " dist = " << dist << endl;

		//cin.get();
		//cout << "diest = " << dist << endl;
		maxVal = maxVal > dist ? maxVal : dist;
		outpMat.at<uchar>(i, j) = (int)(dist*255/SUB_IMG);
		}
	//Mat outpMatNormalizex = Mat::zeros(height, width, CV_8U);
	//outpMatNormalizex = outpMat * 255 / maxVal;
	//outpMat = outpMat * 255 / maxVal;
	Mat outpMatColor = Mat::zeros(height, width, CV_32F);
	applyColorMap(outpMat, outpMatColor, COLORMAP_OCEAN);
	imwrite(img_name, outpMatColor);
}


void VisualizeCC::drawCorrelationHeatMap(vector<vector<OutpStr>>data, string img_name)
{

	int height = data.size();
	int width = data[0].size();
	Mat outpMat = Mat::zeros(height, width, CV_8U);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
		OutpStr temp = data[i][j];
	
		float corr = temp.m_CCcoeff;
		int norm_corr = (corr + 1.0) / 2.0*255.0;
		outpMat.at<uchar>(i, j) = norm_corr;
		}
	Mat outpMatColor = Mat::zeros(height, width, CV_32F);
	applyColorMap(outpMat, outpMatColor, COLORMAP_OCEAN);
	imwrite(img_name, outpMatColor);
}


void VisualizeCC::drawDirectionHeatMap(vector<vector<OutpStr>>data, string img_name)
{
	int height = data.size();
	int width = data[0].size();
	Mat hsv(height, width, CV_8UC3);
	cvtColor(hsv, hsv, CV_RGB2HSV);

	int max_dist = (SUB_IMG - PATCH_SIZE)/2;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
		OutpStr temp = data[i][j];

		int dx = temp.m_point2.x - temp.m_point1.x;
		int dy = temp.m_point2.y - temp.m_point1.y;
		float dist = sqrtf(pow(dx, 2) + pow(dy, 2));

		float angle = atan2(dy, dx) * 180 / 3.14;
		if (angle < 0)
			angle = 360 + angle;

		Vec3b tempHSV;
		tempHSV.val[0] = angle/2;
		tempHSV.val[1] = (int)(dist * 255 / max_dist);
		tempHSV.val[2] = 255;


		hsv.at<Vec3b>(i, j) = tempHSV;
		}
	cvtColor(hsv, hsv, CV_HSV2RGB);
	imwrite(img_name, hsv);
}
