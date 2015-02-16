#ifndef params
#define params
#include <string>

const std::string image1name = "img/ref1.tif";
const std::string image2name = "img/skrecenie.tif";
static const int IMAGE_WIDTH_ = 1500;//szerokosc obrazu
static const int IMAGE_HEIGHT_ = 1500;//wysokosc obrazu


static const int PATCH_SIZE = 15;// rozmiar patcha
static const int SUB_IMG = 25;// rozmiar obszaru przeszukiwan

static const int RESIZE_COEFF = 1; // wspolczynnik skalowania w interpolacji

static const int IMAGE_WIDTH = IMAGE_WIDTH_*RESIZE_COEFF;
static const int IMAGE_HEIGHT = IMAGE_HEIGHT_*RESIZE_COEFF;
#endif