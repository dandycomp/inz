#ifndef params
#define params
#include <string>

#if 0
const std::string image1name = "img/test/ref1.tif";
const std::string image2name = "img/test/ref1.tif";// "img/test/skrecenie1.tif";
static const int IMAGE_WIDTH_ = 1500;//szerokosc obrazu
static const int IMAGE_HEIGHT_ = 1500;//wysokosc obrazu
#endif


#if 0

const std::string image1name = "img/test1/ref.jpg";
//const std::string image2name = "img/test1/ref_edited.jpg";
const std::string image2name = "img/test1/wave.jpg";
static const int IMAGE_WIDTH_ = 1000;//szerokosc obrazu
static const int IMAGE_HEIGHT_ = 1000;//wysokosc obrazu

#endif


#if 0
const std::string image1name = "img/2500_1.tif";
const std::string image2name = "img/2500_1_smudge.tif";
static const int IMAGE_WIDTH_ = 2500;//szerokosc obrazu
static const int IMAGE_HEIGHT_ = 2500;//wysokosc obrazu
#endif


#if 1
const std::string image1name = "img/test1/2/ref.jpg";
const std::string image2name = "img/test1/2/img_edited.jpg";
static const int IMAGE_WIDTH_ = 1000;//szerokosc obrazu
static const int IMAGE_HEIGHT_ = 1000;//wysokosc obrazu
#endif


static const int PATCH_SIZE = 25;// rozmiar patcha
static const int SUB_IMG = 35;// rozmiar obszaru przeszukiwan

//static const int PATCH_SIZE = 13;// rozmiar patcha
//static const int SUB_IMG = 15;// rozmiar obszaru przeszukiwan
static const int RESIZE_COEFF = 3; // wspolczynnik skalowania w interpolacji

static const int IMAGE_WIDTH = IMAGE_WIDTH_*RESIZE_COEFF;
static const int IMAGE_HEIGHT = IMAGE_HEIGHT_*RESIZE_COEFF;
#endif