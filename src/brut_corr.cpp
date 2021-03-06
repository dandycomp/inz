/*
 * brut_corr.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: ivan
 */

#include "brut_corr.hpp"

void NCC::setZero()
{
	m_img1 =  Mat::zeros(1,1,CV_8U);
	m_img2 =  Mat::zeros(1,1,CV_8U);
	m_patch  = 0;
	m_subImg = 0;
	m_corrImg = Mat::zeros(1,1,CV_8U);
}

void NCC::setParameters(Mat& img1, Mat& img2,
			size_t subImg, size_t patch)
{
	if(!img1.data || !img2.data ||
			img1.channels() != img2.channels() ||
			img1.size() != img2.size())
		{
			cout << "\n Error of one of three reasons: "
					"\n1.Can't load images. "
					"\n2.Different nr of channels. "
					"\n3.Different size." << endl;
			setZero();
			cout << " Press any key" << endl;
			cin.get();
			return;
		}

	if(img1.channels() != 1 && img2.channels()!= 1 )
	{
		cvtColor(img1, m_img1, CV_RGB2GRAY);
		cvtColor(img2, m_img2, CV_RGB2GRAY);
	}
	else
	{
		m_img1 = img1;
		m_img2 = img2;
	}

	if( (patch%2 == 1 && subImg%2 == 1)
			|| ( patch > 2 && subImg > 4 ))
	{
		m_patch = patch;
		m_subImg = subImg;
	}
	else
	{
		cout << "Wrong patch or SubImg" << endl;
		cout << " Press any key" << endl;
		cin.get();
	}

	m_corrImg = Mat::zeros(img1.rows-subImg+1,
				img1.cols-subImg+1, CV_32F);
}

//return medium float value 0...255
float NCC::getMediumValueAroundPoint(Mat& img, Point pn)
{
	int dist = (m_patch-1) / 2;
	if( ( pn.x > img.cols-dist) ||
		( pn.x - dist < 0 )     ||
		( pn.y > img.rows-dist) ||
		( pn.y - dist < 0 )         )
	{
		cout << " Out of range. Medium function. " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

	float medVal = 0.0;
	int sum = 0;
	int counter = 0;

	for( int i = -dist; i <= dist; i++)
		for( int j = -dist; j <= dist; j++)
		{
			counter++;
			sum += img.at<uchar>(
					Point( pn.x+i,pn.y+j ) );
		}

	medVal = static_cast<float>(sum) / static_cast<float>(counter);
	return medVal;
}


float NCC::getStDeviationValueAroundPoint(Mat& img, Point pn, float mediumValue)
{
	int dist = (m_patch-1) / 2;
	if( ( pn.x > img.cols-dist) ||
		( pn.x - dist < 0 )     ||
		( pn.y > img.rows-dist) ||
		( pn.y - dist < 0 )         )
	{
		cout << " Out of range. GetStdDev function. " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

	float deviation = 0.0;

	for( int i = -dist; i <= dist; i++)
		for( int j = -dist; j <= dist; j++)
		{
			float temp = img.at<uchar>(
					Point( pn.x+i,pn.y+j ));

			temp -= mediumValue;
			deviation += temp*temp;
		}

    float smallEpsilon = 0.00001;
    deviation = abs(deviation) > smallEpsilon ? deviation : smallEpsilon;
	return deviation;
}

float NCC::getStDeviationValueAroundPoint(Mat& img, Point pn)
{
	float medValue = getMediumValueAroundPoint(img, pn);
	return getStDeviationValueAroundPoint(img, pn, medValue);
}

float NCC::getCorrelate(Point p1, Point p2)
{
	int dist = (m_patch-1) / 2;
	int maxX = max(p1.x, p2.x);
	int maxY = max(p1.y, p2.y);
	int minY = min(p1.y, p2.y);
	int minX = min(p1.x, p2.x);

	// we are sure that m_img1.size() == m_img2.size()
	if( ( maxX > m_img1.cols-dist-1) ||
		( minX - dist < 0 )     ||
		( maxY > m_img1.rows-dist-1)||
		( minY - dist < 0 )         )
	{
		cout << " Out of range of image, chose other point. GetCorrelate(p1,p2) " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

	float numerator   = 0.0;
	float denominator = 0.0;

	float med1   = getMediumValueAroundPoint(m_img1, p1);
	float stDev1 = getStDeviationValueAroundPoint(m_img1, p1, med1);

	float med2   = getMediumValueAroundPoint(m_img2, p2);
	float stDev2 = getStDeviationValueAroundPoint(m_img2, p2, med2);

	denominator = sqrtf( stDev1 * stDev2 );
	int counter = 0;

	for( int i = -dist; i <= dist; i++)
		for( int j = -dist; j <= dist; j++)
		{
				float temp1 = m_img1.at<uchar>(
						Point( p1.x+i,p1.y+j ) );
				float temp2 = m_img2.at<uchar>(
						Point( p2.x+i,p2.y+j ) );


				numerator   += ( temp1 - med1 ) *
							   ( temp2 - med2 );
				counter++;
		}

	if (numerator == 0)
		return 1;
	

	return numerator/denominator;
}

OutpStr NCC::getBestCorrFromArea(Point pnt)
{
	int dist = (m_subImg-1) / 2;
	if( ( pnt.x >= m_img1.cols-dist) ||
		( pnt.x - dist < 0 )     ||
		( pnt.y >= m_img1.rows-dist) ||
		( pnt.y - dist < 0 )         )
	{
		cout << " Out of range. getBestCorrFromArea function. " << endl;
		cout << "Pnt = " << pnt << endl;
		cout << " Press any key" << endl;
		cin.get();
		return OutpStr();
	}

	float bestCorrelation = -2;
	Point p_deform = Point(0, 0);
	int tempRange = ( m_subImg - m_patch ) / 2;

	if (getCorrelate(pnt, pnt) == 1)
	{
		OutpStr outp(pnt, pnt, 1);
		return outp;
	}

	//search subImg region with patch
	for(int i = -tempRange; i <= tempRange; i++ )
		for(int j = -tempRange; j<=tempRange; j++)
		{
		Point p_2 = Point(pnt.x + i, pnt.y + j);
		float tempCorrelate = getCorrelate(pnt, p_2);

			if (tempCorrelate > bestCorrelation)
			{
				bestCorrelation = tempCorrelate;
				p_deform = p_2;
			}
		}

	OutpStr outp(pnt, p_deform, bestCorrelation);
	return outp;
}

void NCC::correlate()
{
	int newNrOfRows = m_img1.rows - m_subImg + 1;
	int newNrOfCols = m_img1.cols - m_subImg + 1;

	Mat corrMat = Mat::zeros(newNrOfRows, newNrOfCols, CV_32F);
	vector< vector< OutpStr >> results;
	vector<OutpStr> col;
	int finishedPercent = 0;
	cout << " \nWyznaczenie cross korelacji metoda NCC, ukonczono: " << endl;

	for (int j = 0; j < newNrOfRows; j++){
		col.clear();
		for (int i = 0; i < newNrOfCols; i++)
		{
			Point tempPnt = Point(i + m_subImg / 2, j + m_subImg / 2);
			OutpStr corr = getBestCorrFromArea(tempPnt);
			col.push_back(corr);

			int actual = (j*newNrOfCols + i) * 100
				/ (newNrOfRows*newNrOfCols);

			if(actual > finishedPercent)
			{
				finishedPercent = actual;
				cout << finishedPercent << " % " << endl;
			}
		}
		results.push_back(col);
	}

	VisualizeCC vcc;
	vcc.drawDirectionHeatMap(results, "img/NCC.jpg");
}