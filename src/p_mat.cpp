#include "p_mat.hpp"
#include "utils.hpp"

Pmat::Pmat(Pmat& pm){

    pm.m_Pmat.copyTo(m_Pmat);//pozniej mozna pomyslec nad referencjami
    m_subImg = pm.m_subImg;
    m_patch  = pm.m_patch;
    m_pnt    = pm.m_pnt;
}

void Pmat::setParameters(Point pnt,
                    int subImg, int patch)
{
    //NIE MA ZADNYCH SPRAWDZEN, TRZEBA POPRAWIC
    m_pnt = pnt;
    m_patch = patch;
    m_subImg = subImg;
}

int Pmat::getPFromImage(Mat& img1, Mat& img2, Point p1, Point p2)
{
    int p_val = 0;
    for(int i = -m_patch/2; i <= m_patch/2; i++)
        for(int j = -m_patch/2; j <= m_patch/2; j++)
        {
            int first  = img1.at<uchar>(p1.y+j, p1.x+i);
            int second = img2.at<uchar>(p2.y+j, p2.x+i);
            p_val += first * second;
        }
    return p_val;
}

void Pmat::writePmat()
{
cout << m_Pmat << endl;
}
// patch oraz subimg brane ze skladowych klasy
void Pmat::setPMat(Mat& img1, Mat& img2)
{
    int sizeOfMat = m_subImg - m_patch + 1;
    m_Pmat = Mat::zeros(sizeOfMat, sizeOfMat, CV_32F);

    int tempRange = ( m_subImg - m_patch )/2;

    for(int i = -tempRange; i <= tempRange; i++ )
		for(int j = -tempRange; j<= tempRange; j++)
		{
            m_Pmat.at<float>(Point(i+tempRange,j+tempRange)) =
                                getPFromImage(img1, img2,
                                m_pnt, Point(m_pnt.x+i, m_pnt.y+j));
		}
}

//get value fro, m_Pmat in (i,j) position
int Pmat::getPFromMat(int i, int j)
{
    return (int)m_Pmat.at<float>(j,i);
}
//do poprawki

int  Pmat::next_P_x(int col, int row, Mat& img1, Mat& img2)
{

int sizeOfMat = m_subImg - m_patch + 1;
int oldPVal = m_Pmat.at<float>(row,col);
int I = 0;
int J = 0;
Point p_nextXFirst = Point(m_pnt.x+1, m_pnt.y);
Point p_nextXSecond = Point(p_nextXFirst.x+col-sizeOfMat/2,
                            p_nextXFirst.y+row-sizeOfMat/2);

//Counting of I argument. Equation (26)
for(int i = -m_patch/2; i <= m_patch/2; i++)
    {
        int first  = img1.at<uchar>(p_nextXFirst.y+i, p_nextXFirst.x+m_patch/2);
        int second = img2.at<uchar>(p_nextXSecond.y+i, p_nextXSecond.x+m_patch/2);
        //cout << "I first,second " << first << "  " << second << endl;
        I += first*second;
    }
//Counting of J argument. Equation (27)
for(int i = -m_patch/2; i <= m_patch/2; i++)
    {
        // -1 na koncu oznacza ze odejmujemy od starego punkta, a nie od nowego
        int first  = img1.at<uchar>(p_nextXFirst .y+i, p_nextXFirst .x-m_patch/2-1);
        int second = img2.at<uchar>(p_nextXSecond.y+i, p_nextXSecond.x-m_patch/2-1);
       // cout << "J first,second " << first << "  " << second << endl;
        J += first*second;
    }
//cout << " \nPoint:" << Point(col, row) << " Values " << oldPVal<< " " << I << " " << J << endl;
return oldPVal + I - J;
}


int  Pmat::next_P_y(int col, int row, Mat& img1, Mat& img2)
{
    int sizeOfMat = m_subImg - m_patch + 1;
    int oldPVal = m_Pmat.at<float>(row,col);
    int I = 0;
    int J = 0;
    Point p_nextYFirst = Point(m_pnt.x, m_pnt.y+1);
    Point p_nextYSecond = Point(p_nextYFirst.x+col-sizeOfMat/2,
                                p_nextYFirst.y+row-sizeOfMat/2);

//cout << "Points: " << p_nextYFirst << " " << p_nextYSecond << endl;
//Counting of I argument. Equation (26)
for(int i = -m_patch/2; i <= m_patch/2; i++)
    {
        int first  = img1.at<uchar>(p_nextYFirst .y+m_patch/2, p_nextYFirst .x+i);
        int second = img2.at<uchar>(p_nextYSecond.y+m_patch/2, p_nextYSecond.x+i);
        //cout << "I first,second " << first << "  " << second << endl;
        I += first*second;
    }
//Counting of J argument. Equation (27)
for(int i = -m_patch/2; i <= m_patch/2; i++)
    {
        // -1 na koncu oznacza ze odejmujemy od starego punkta, a nie od nowego
        int first  = img1.at<uchar>(p_nextYFirst .y-m_patch/2-1,  p_nextYFirst .x+i);
        int second = img2.at<uchar>(p_nextYSecond.y-m_patch/2-1,  p_nextYSecond.x+i);
        J += first*second;
    }

    return oldPVal + I - J;
}

Pmat Pmat::next_x_Pmat(Mat& img1, Mat& img2)
{
    if(m_pnt.x >= img1.cols)
    {
        cout << " Out of range.nextXfun" << endl;
        cout << " Press any key" << endl;
        cin.get();
    }
    Pmat nextX;
    nextX.setParameters(Point(m_pnt.x+1, m_pnt.y), m_subImg, m_patch);
    int sizeOfMat = m_subImg - m_patch + 1;
    nextX.m_Pmat = Mat::zeros(sizeOfMat, sizeOfMat, CV_32F);

    int tempRange = ( m_subImg - m_patch ) / 2;

    int diff = m_subImg - m_patch;

    for(int i = 0; i <= diff; i++)
        for(int j =0; j <= diff; j++)
        {
            nextX.m_Pmat.at<float>(j,i) = next_P_x(i,j, img1, img2);
        }
    return nextX;
}
//do poprawki
Pmat Pmat::next_y_Pmat(Mat& img1, Mat& img2)
{

    if(m_pnt.y >= img1.rows)
    {
        cout << " Out of range.nextYfun" << endl;
        cout << " Press any key" << endl;
        cin.get();
    }
    Pmat nextY;
    nextY.setParameters(Point(m_pnt.x, m_pnt.y+1), m_subImg, m_patch);
    int sizeOfMat = m_subImg - m_patch + 1;
    nextY.m_Pmat = Mat::zeros(sizeOfMat, sizeOfMat, CV_32F);

    int tempRange = ( m_subImg - m_patch ) / 2;

    int diff = m_subImg - m_patch;

    for(int i = 0; i <= diff; i++)
        for(int j =0; j <= diff; j++)
        {
            nextY.m_Pmat.at<float>(j,i) = next_P_y(i,j, img1, img2);
        }
    return nextY;
}
