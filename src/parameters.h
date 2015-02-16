#ifndef params
#define params
#include <string>

const std::string image1name = "img/test1/2/ref.jpg";
const std::string image2name = "img/test1/2/img_edited.jpg";
static const int IMAGE_WIDTH_ = 1000;//szerokosc obrazu
static const int IMAGE_HEIGHT_ = 1000;//wysokosc obrazu


static const int PATCH_SIZE = 25;// rozmiar patcha
static const int SUB_IMG = 35;// rozmiar obszaru przeszukiwan

//static const int PATCH_SIZE = 13;// rozmiar patcha
//static const int SUB_IMG = 15;// rozmiar obszaru przeszukiwan
static const int RESIZE_COEFF = 3; // wspolczynnik skalowania w interpolacji

static const int IMAGE_WIDTH = IMAGE_WIDTH_*RESIZE_COEFF;
static const int IMAGE_HEIGHT = IMAGE_HEIGHT_*RESIZE_COEFF;
#endif