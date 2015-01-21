#ifndef params
#define params
#include <string>

#if 1
const std::string image1name = "img/HC Reference.tif";
const std::string image2name = "img/HCY00 X00.tif";
static const int IMAGE_WIDTH_ = 485;
static const int IMAGE_HEIGHT_ = 327;
#endif

#if 0
const std::string image1name = "img/col_3.jpg";
const std::string image2name = "img/col_3.jpg";
static const int IMAGE_WIDTH_ = 324;
static const int IMAGE_HEIGHT_ = 240;
#endif

#if 0
const std::string image1name = "img/col_15.jpg";
const std::string image2name = "img/col1_15.jpg";
static const int IMAGE_WIDTH_ = 120;
static const int IMAGE_HEIGHT_ = 90;
#endif


#if 0
const std::string image1name = "img/col.jpg";
const std::string image2name = "img/col1.jpg";
static const int IMAGE_WIDTH_ = 133;
static const int IMAGE_HEIGHT_ = 98;
#endif

#if 0
const std::string image1name = "img/col_small.jpg";
const std::string image2name = "img/col1_small.jpg";
static const int IMAGE_WIDTH_ = 54;
static const int IMAGE_HEIGHT_ = 40;
#endif

#if 0
const std::string image1name = "img/nature1_.jpg";
const std::string image2name = "img/nature2_.jpg";
static const int IMAGE_WIDTH_ = 1500;
static const int IMAGE_HEIGHT_ = 800;
#endif

static const int PATCH_SIZE = 7;// rozwiazac ten problem za posrednictwem parametrow wejsciowych
static const int SUB_IMG = 15;
static const int RESIZE_COEFF = 5;


static const int IMAGE_WIDTH = IMAGE_WIDTH_*RESIZE_COEFF;
static const int IMAGE_HEIGHT = IMAGE_HEIGHT_*RESIZE_COEFF;
//static const int PATCH_SIZE = PATCH_SIZE_*RESIZE_COEFF;// rozwiazac ten problem za posrednictwem parametrow wejsciowych
//static const int SUB_IMG = SUB_IMG_*RESIZE_COEFF;


//static const int IMAGE_WIDTH = 1366;
//static const int IMAGE_HEIGHT = 768;

//const string image1name = "img/img1.jpg";
//const string image2name = "img/img2.jpg";



#endif