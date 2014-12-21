#include "fast_cc.hpp"
#include "utils.hpp"
#include <fstream>

void FastCC::setParameters( GlSumTbl& gl1, GlSumTbl& gl2 )
{
    if(!gl1.data() || !gl2.data())
    {
        cout << " Wrong data, setParameters function" << endl;
        cout << " Press any key." << endl;
        cin.get();
    }
    m_Fmat = Mat::zeros(gl1.m_glSum.size(), CV_32F);
    m_Gmat = Mat::zeros(gl1.m_glSum.size(), CV_32F);
    m_gl1 = gl1;
    m_gl2 = gl2;
    m_subImg = gl1.m_subImg;
    m_patch  = gl1.m_patch;

// mozna wyrzucic do oddzielnej funcji spradzajac spelnienia warunkow
//pamietajmy o tym, ze zero tez jest punktem, tak ze nie dodajemy +1
    Point startPoint = Point(m_subImg/2, m_subImg/2);

    m_pVal.setParameters(startPoint, m_subImg, m_patch);
    m_pVal.setPMat(m_gl1.m_img, m_gl2.m_img);

    m_nextXpVal = m_pVal.next_x_Pmat(m_gl1.m_img, m_gl2.m_img);
    //m_nextXpVal.writePmat();
    m_nextYpVal = m_pVal.next_y_Pmat(m_gl1.m_img, m_gl2.m_img);
    //m_nextYpVal.writePmat();
}

//equation (12) and (13)
void FastCC::create_F_and_G_mat()
{
    float nrOfpixelsInPatch = (float)m_patch*m_patch;
    for(int i = 0; i < m_Fmat.cols; i++)
        for(int j = 0; j < m_Fmat.rows; j++){
            m_Fmat.at<float>(j,i) = m_gl1.m_glSqSum.at<float>(j,i) -
                        m_gl1.m_glSum.at<float>(j,i)*
                        m_gl1.m_glSum.at<float>(j,i)
                        / nrOfpixelsInPatch;

            m_Gmat.at<float>(j,i) = m_gl2.m_glSqSum.at<float>(j,i) -
                        m_gl2.m_glSum.at<float>(j,i)*m_gl2.m_glSum.at<float>(j,i)
                        / nrOfpixelsInPatch;
        }
}

int FastCC::getP( Point p1, Point p2 )
{
    int sum = 0;
    int dist = m_patch / 2;

    for( int i = -dist; i <= dist; i++)
		for( int j = -dist; j <= dist; j++)
		{
            Point p1_ = Point(p1.x+i, p1.y+j);
            Point p2_ = Point(p2.x+i, p2.y+j);

           sum += m_gl1.m_img.at<uchar>(p1_) * m_gl2.m_img.at<uchar>(p2_);
        }
    return sum;
}

//P(col,row)
float FastCC::getQ( Point p1, Point p2 )
{
    float nrOfpixelsInPatch = (float)m_patch*m_patch;
    int dist = m_patch / 2;

    p1 = Point(p1.x-dist, p1.y-dist);
    p2 = Point(p2.x-dist, p2.y-dist);

    return m_gl1.m_glSum.at<float>(p1) *
           m_gl2.m_glSum.at<float>(p2) / nrOfpixelsInPatch;
}

float FastCC::getF( Point pnt)
{
    int dist = m_patch / 2;
    pnt = Point(pnt.x-dist, pnt.y-dist);
    return m_Fmat.at<float>(pnt);
}

float FastCC::getG( Point pnt)
{
    int dist = m_patch / 2;
    pnt = Point(pnt.x-dist, pnt.y-dist);
    return m_Gmat.at<float>(pnt);
}

float FastCC::getDist(Point p1, Point p2)
{
    float distX = p2.x - p1.x;
    float distY = p2.y - p1.y;
    return sqrt( distX*distX + distY*distY );
}

float FastCC::getCorrelate(Point p1, Point p2)
{
    int dist = m_patch / 2;
    int maxX = max(p1.x, p2.x);
	int maxY = max(p1.y, p2.y);
	// we are sure that m_img1.size() == m_img2.size()
	if( ( maxX > m_gl1.m_img.cols-dist) ||
		( maxX - dist < 0 )     ||
		( maxY > m_gl1.m_img.rows-dist)||
		( maxY - dist < 0 )         )
	{
		cout << " Out of range of image, chose other point. GetCorrelate(p1,p2) " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

    float   Q = getQ(p1,p2);
    float   P = (float)getP(p1,p2);

    float F = getF(p1);
    float G = getG(p2);

    //cout << "\n In FAST points " << p1 << " and " << p2 << endl;
    //cout << " P= " << P << " Q= " << Q << " diff = " << P-Q <<  " F= " << F<< " G= "<< G << endl;
    return (P-Q)/sqrt(F*G);
}

float FastCC::getRecursiveCorrelate(Point p1, Point p2, float pVal)
{
    int dist = m_patch / 2;
    int maxX = max(p1.x, p2.x);
	int maxY = max(p1.y, p2.y);
	// we are sure that m_img1.size() == m_img2.size()
	if( ( maxX > m_gl1.m_img.cols-dist) ||
		( maxX - dist < 0 )     ||
		( maxY > m_gl1.m_img.rows-dist)||
		( maxY - dist < 0 )         )
	{
		cout << " Out of range of image, chose other point. GetCorrelate(p1,p2) " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

    float   Q = getQ(p1,p2);
    float F = getF(p1);
    float G = getG(p2);

    //cout << "\n In Recursive points " << p1 << " and " << p2 << endl;
    //cout << " P= " << pVal << " Q= " << Q << " diff = " << pVal-Q <<  " F= " << F<< " G= "<< G << endl;
    return (pVal-Q)/sqrt(F*G);
}

float FastCC::getBestCorrFromArea(Point pnt)
{

	int dist = (m_subImg-1) / 2;
	if( ( pnt.x > m_gl1.m_img.cols-dist) ||
		( pnt.x - dist < 0 )     ||
		( pnt.y > m_gl1.m_img.rows-dist) ||
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
		}
	return bestCorrelation;
}

float FastCC::getRecursiveBestCorrFromArea(Point pnt, Pmat pmat)
{
	int dist = (m_subImg-1) / 2;
	if( ( pnt.x > m_gl1.m_img.cols-dist) ||
		( pnt.x - dist < 0 )     ||
		( pnt.y > m_gl1.m_img.rows-dist) ||
		( pnt.y - dist < 0 )         )
	{
		cout << " Out of range. getBestCorrFromArea function. " << endl;
		cout << " Press any key" << endl;
		cin.get();
		return 0;
	}

	float bestCorrelation = -2.0;
	int tempRange = ( m_subImg - m_patch ) / 2;
	for(int i = -tempRange; i <= tempRange; i++ )
		for(int j = -tempRange; j<=tempRange; j++)
		{

            int p_val = pmat.getPFromMat(i+tempRange, j+tempRange);

			float tempCorrelate = getRecursiveCorrelate(pnt,Point(pnt.x+i, pnt.y+j), p_val);

			bestCorrelation = bestCorrelation >
							tempCorrelate ? bestCorrelation:tempCorrelate;

			if(bestCorrelation == 1)
				return bestCorrelation;
		}
	return bestCorrelation;
}

void FastCC::runFastCC()
{
    create_F_and_G_mat();
    int newNrOfRows = m_gl1.m_img.rows - m_subImg + 1;
	int newNrOfCols = m_gl1.m_img.cols - m_subImg + 1;

	Mat corrMat = Mat::zeros(newNrOfRows, newNrOfCols, CV_8U);
	int finishedPercent = 0;

	for(int j = 0; j < newNrOfRows; j++)
        for(int i = 0; i < newNrOfCols; i++)
		{
			Point tempPnt = Point(i+m_subImg/2,j+m_subImg/2);

			float corr = getBestCorrFromArea(tempPnt);

            int norm_corr = (corr + 1.0)/2.0*255.0;

			corrMat.at<uchar>(j,i) = norm_corr;
			int actual = (j*newNrOfCols+i)*100
						/(newNrOfRows*newNrOfCols);

			if(actual > finishedPercent)
			{
				finishedPercent = actual;
				cout << finishedPercent << " % " << endl;
			}
		}
    //Mat outp;
    //applyColorMap(corrMat, outp, COLORMAP_OCEAN);
    //imwrite("img/FastCorrelation.jpg", outp);
	imwrite("img/FastCorrelation.jpg", corrMat);
	cout << " End of fast" << endl;
}

void FastCC::recursiveFastCC()
{
    create_F_and_G_mat();
    int newNrOfRows = m_gl1.m_img.rows - m_subImg + 1;
	int newNrOfCols = m_gl1.m_img.cols - m_subImg + 1;

	Mat corrMat = Mat::zeros(newNrOfRows, newNrOfCols, CV_8U);
	int finishedPercent = 0;

    for(int j = 0; j < newNrOfRows; j++)
    {
		for(int i = 0; i < newNrOfCols; i++)
		{
			Point tempPnt = Point(i+m_subImg/2,j+m_subImg/2);

            float corr = getRecursiveBestCorrFromArea(tempPnt, m_pVal);
            int norm_corr = (corr + 1.0)/2.0*255.0;
			corrMat.at<uchar>(j,i) = norm_corr;
			int actual = (j*newNrOfCols+i)*100
						/(newNrOfRows*newNrOfCols);

			if(actual > finishedPercent)
			{
				finishedPercent = actual;
				cout << finishedPercent << " % " << endl;
			}
           m_pVal = m_nextXpVal;
		   //cout << " 1 " << endl;
           m_nextXpVal = m_pVal.next_x_Pmat(m_gl1.m_img, m_gl2.m_img);
		   //cout << " 2 " << endl;
		}
		
		m_pVal = m_nextYpVal;
        m_nextYpVal = m_pVal.next_y_Pmat(m_gl1.m_img, m_gl2.m_img);
        m_nextXpVal = m_pVal.next_x_Pmat(m_gl1.m_img, m_gl2.m_img);
    }

    //normalize(tempMat, corrMat, 0, 255, NORM_MINMAX, CV_8U);

    //Mat outp;
    //applyColorMap(corrMat, outp, COLORMAP_OCEAN);
	
	imwrite("img/FastRecursiveCorrelation.jpg", corrMat);
	//imwrite("img/FastRecursiveCorrelation.jpg", corrMat);

    //float tt = getDist(Point(10,10), Point(12, 16));
    cout << " End of fast recursive " << endl;

}

