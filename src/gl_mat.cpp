
#include "gl_mat.hpp"
#include "utils.hpp"

using namespace cv;

void GlSumTbl::setZero()
{
	m_img =  Mat::zeros(1,1,CV_8U);
	m_glSum =  Mat::zeros(1,1,CV_8U);
	m_glSqSum = Mat::zeros(1,1,CV_8U);
	m_patch  = 0;
	m_subImg = 0;
}

bool GlSumTbl::data()
{
    if(m_img.data && m_glSqSum.data
    && m_glSum.data && m_patch > 2 &&
    m_subImg > 4)
    {
        return true;
    }
    return false;
}

GlSumTbl::GlSumTbl(GlSumTbl& gst)
{
    m_img = gst.m_img;
    m_glSum = gst.m_glSum;
    m_glSqSum = gst.m_glSqSum;
    m_patch = gst.m_patch;
    m_subImg = gst.m_subImg;
}

void GlSumTbl::setParameters(Mat& img ,int subImg, int patch)
{

	if(!img.data)
		{
			cout << "\n Can't load image. SetParameters" << endl;
		//	setZero();
			cout << " Press any key" << endl;
			cin.get();
			return;
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

	if(img.channels() != 1)
	{
		cvtColor(img, m_img, CV_RGB2GRAY);
	}
	else
	{
		m_img = img;
	}

//create an zero Mat of size img.size - patch + 1
	m_glSum = Mat::zeros(img.rows-patch+1,
				img.cols-patch+1, CV_32F);

    m_glSqSum = Mat::zeros(img.rows-patch+1,
                img.cols-patch+1, CV_32F);

    m_subImg = subImg;
    m_patch = patch;
}

//get sum of pixels values around pn in patch area
int GlSumTbl::getSumFromIntegral(Mat &sumIntgrl, Point pnt)
{
	pnt = Point(pnt.x, pnt.y);//because of displacement 1
	int halOfPatch = ( m_patch -1 ) / 2;

	Point p1 = Point(pnt.x - halOfPatch - 1, pnt.y - halOfPatch - 1);// -1 jest potrzebne, nie usuwac
	Point p2 = Point(pnt.x + halOfPatch, pnt.y + halOfPatch);

	if( p1.x < 0 || p1.y < 0 )
        {
       // cout << " p1.x < 0 || p1.y < 0  " << endl;
        //cout << " Press any key" << endl;
		//cin.get();

            int sum = 0;
            sum += sumIntgrl.at<int>(p2.y, p2.x);

            if(p1.x >= 0)
                {sum -= sumIntgrl.at<int>(p2.y, p1.x);}

            else if(p1.y >= 0)
                {sum -= sumIntgrl.at<int>(p1.y, p2.x);}
            //cout << " Sum = " << sum << endl;
            return sum;
        }
//cout << "\n " << sumIntgrl.at<int>(p2) << " " << sumIntgrl.at<int>(p2.y, p1.x) << " " <<
//sumIntgrl.at<int>(p1.y, p2.x) << " " << sumIntgrl.at<int>(p1) << endl;
//cin.get();

	 return sumIntgrl.at<int>(p2.y, p2.x)
          - sumIntgrl.at<int>(p2.y, p1.x)
		  -	sumIntgrl.at<int>(p1.y, p2.x)
          + sumIntgrl.at<int>(p1.y, p1.x);
}

//return integralimage of type CV_32F with int type in each position
void GlSumTbl::toIntegral(Mat& sumInt, Mat& sqSumInt)
{
#if 0
    Mat newSum, newSqSum;
    integral(m_img, newSum, newSqSum);
    newSum( Rect(Point(1,1), Point(m_img.cols, m_img.rows)) ).copyTo(sumInt);
    newSqSum( Rect(Point(1,1), Point(m_img.cols, m_img.rows))).copyTo(sqSumInt);
    //utils::exportMat<int>(sumInt  , "Mat/integralSum.txt");
#endif


    Mat sumIntegral   = Mat::zeros(m_img.size(), CV_32F);
    Mat sqSumIntegral = Mat::zeros(m_img.size(), CV_32F);

    int zero_zero = m_img.at<uchar>(0,0);
	sumIntegral  .at<int>(Point(0,0)) = zero_zero;
	sqSumIntegral.at<int>(Point(0,0)) = zero_zero*zero_zero;
	//fill first row

	for(int i = 1; i < m_img.rows; i++)
	{
        int row = m_img.at<uchar>(i,0);
		sumIntegral  .at<int>(i,0) = row     + sumIntegral  .at<int>(i-1,0);
		sqSumIntegral.at<int>(i,0) = row*row + sqSumIntegral.at<int>(i-1,0);
	}

	//fill first column
	for(int i = 1; i < m_img.cols; i++)
	{
        int col = m_img.at<uchar>(0,i);
		sumIntegral  .at<int>(0,i) = col     + sumIntegral  .at<int>(0, i-1);
		sqSumIntegral.at<int>(0,i) = col*col + sqSumIntegral.at<int>(0, i-1);
	}

	for(int i = 1; i < m_img.rows; i++)
		for( int j = 1; j < m_img.cols; j++ )
		{
            int next = m_img.at<uchar>(i,j);
			sumIntegral.at<int>(i,j) =    next + sumIntegral.at<int>(i,j-1)+
					sumIntegral.at<int>(i-1,j) - sumIntegral.at<int>(i-1,j-1);

            sqSumIntegral.at<int>(i,j) =next*next + sqSumIntegral.at<int>(i,j-1)+
                     sqSumIntegral.at<int>(i-1,j) - sqSumIntegral.at<int>(i-1,j-1);
		}
    sumIntegral.copyTo  (sumInt);
	sqSumIntegral.copyTo(sqSumInt);
}

void GlSumTbl::createGlobalSum(Mat &sumIntgrl, Mat &sqSumIntgrl)
{
    int halOfPatch = ( m_patch -1 ) / 2;
    for(int i = 0; i < m_glSum.cols; i++)
        for(int j=0; j < m_glSum.rows; j++)
        {
           float temp1 = getSumFromIntegral(sumIntgrl   , Point(i+halOfPatch,j+halOfPatch));
           float temp2 = getSumFromIntegral(sqSumIntgrl , Point(i+halOfPatch,j+halOfPatch));
           m_glSum  .at<float>(j,i) = temp1;
           m_glSqSum.at<float>(j,i) = temp2;
        }
}

void GlSumTbl::createGlSums()
{
    Mat sumIntegral;
    Mat sqSumIntegral;
    toIntegral(sumIntegral, sqSumIntegral);
    createGlobalSum(sumIntegral, sqSumIntegral);
}
// przejrzeć wszystkie kody na przesunięcia.
