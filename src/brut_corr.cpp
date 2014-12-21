/*
 * brut_corr.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: ivan
 */

#include "brut_corr.hpp"

void BrutCorr::setZero()
{
	m_img1 =  Mat::zeros(1,1,CV_8U);
	m_img2 =  Mat::zeros(1,1,CV_8U);
	m_patch  = 0;
	m_subImg = 0;
	m_corrImg = Mat::zeros(1,1,CV_8U);
}

void BrutCorr::setParameters(Mat& img1, Mat& img2,
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
float BrutCorr::getMediumValueAroundPoint(Mat& img, Point pn)
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


float BrutCorr::getStDeviationValueAroundPoint(Mat& img, Point pn, float mediumValue)
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

float BrutCorr::getStDeviationValueAroundPoint(Mat& img, Point pn)
{
	float medValue = getMediumValueAroundPoint(img, pn);
	return getStDeviationValueAroundPoint(img, pn, medValue);
}

float BrutCorr::getCorrelate(Point p1, Point p2)
{
	int dist = (m_patch-1) / 2;
	int maxX = max(p1.x, p2.x);
	int maxY = max(p1.y, p2.y);
	// we are sure that m_img1.size() == m_img2.size()
	if( ( maxX > m_img1.cols-dist) ||
		( maxX - dist < 0 )     ||
		( maxY > m_img1.rows-dist)||
		( maxY - dist < 0 )         )
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
                //cout << Point( p1.x+i,p1.y+j ) << endl;
				float temp2 = m_img2.at<uchar>(
						Point( p2.x+i,p2.y+j ) );

			//	cout << " temps " << temp1 << "  " << temp2 << endl;
				numerator   += ( temp1 - med1 ) *
							   ( temp2 - med2 );
			//	cout << "num = " << numerator << endl;
				counter++;
		}

	//cout << "\nNumerator: " << numerator <<" denominator: " << denominator <<
	//		" counter = " << counter << endl;
	return numerator/denominator;
}

float BrutCorr::getBestCorrFromArea(Point pnt)
{
	int dist = (m_subImg-1) / 2;
	if( ( pnt.x > m_img1.cols-dist) ||
		( pnt.x - dist < 0 )     ||
		( pnt.y > m_img1.rows-dist) ||
		( pnt.y - dist < 0 )         )
	{
		cout << " Out of range. getBestCorrFromArea function. " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

	float bestCorrelation = -2;
	int tempRange = ( m_subImg - m_patch ) / 2;
	//search subImg region with patch
	for(int i = -tempRange; i <= tempRange; i++ )
		for(int j = -tempRange; j<=tempRange; j++)
		{
			float tempCorrelate = getCorrelate(pnt,Point(pnt.x+i, pnt.y+j));

			bestCorrelation = bestCorrelation >
						tempCorrelate ? bestCorrelation:tempCorrelate;

			if(bestCorrelation == 1)
				return bestCorrelation;

	//		cout << "Corr:" << pnt << " " << Point(pnt.x+i, pnt.y+j) <<
	//				" " << tempCorrelate << " " << bestCorrelation << endl;
		}
	return bestCorrelation;
}

void BrutCorr::correlate()
{
	//float Bestcorr = getBestCorrFromArea(Point(100,100));
	//float corr = getCorrelate(Point(453,353), Point(453,353));
	//float corr = getCorrelate(Point(100,50), Point(100,50));
//	cout << "\n Brutal method correlation = " << corr <<endl;//<< " and best corr = " << Bestcorr << endl;

#if 1
	int newNrOfRows = m_img1.rows - m_subImg + 1;
	int newNrOfCols = m_img1.cols - m_subImg + 1;

	Mat corrMat = Mat::zeros(newNrOfRows, newNrOfCols, CV_32F);
	//cout << " Img size = " << corrMat.size() << endl;
	int finishedPercent = 0;
	//cout << m_subImg << m_subImg/2 << endl;
	for(int i = 0; i < newNrOfCols; i++)
		for(int j = 0; j < newNrOfRows; j++)
		{
			Point tempPnt = Point(i+m_subImg/2,j+m_subImg/2);
			//cout << "Pnt " <<  tempPnt << endl;
			float corr = getBestCorrFromArea(tempPnt);
			corrMat.at<float>(j,i) = (corr+1.0)/2.0;
	//		cout << "2" << endl;
			int actual = (i*newNrOfRows+j)*100
						/(newNrOfRows*newNrOfCols);

			if(actual > finishedPercent)
			{
				finishedPercent = actual;
				cout << finishedPercent << " % " << endl;
			}
		}
    imwrite("img/BrutalCorrelation.jpg", corrMat*255);
#endif

}