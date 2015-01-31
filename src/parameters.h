#ifndef params
#define params
#include <string>

#if 0
const std::string image1name = "img/kot.jpg";
//const std::string image2name = "img/cuted/ref_100.tif";
const std::string image2name = "img/kotX.jpg";
static const int IMAGE_WIDTH_ = 1600;
static const int IMAGE_HEIGHT_ = 1200;
#endif

#if 0
const std::string image1name = "img/cuted/30pix1.tif";
//const std::string image2name = "img/cuted/ref_100.tif";
const std::string image2name = "img/cuted/30pix2.tif";
static const int IMAGE_WIDTH_ = 30;
static const int IMAGE_HEIGHT_ = 30;
#endif

#if 0
const std::string image1name = "img/cuted/ref_100.tif";
//const std::string image2name = "img/cuted/ref_100.tif";
const std::string image2name = "img/cuted/X5Y0_100.tif";
static const int IMAGE_WIDTH_ = 100;
static const int IMAGE_HEIGHT_ = 100;
#endif

#if 0
const std::string image1name = "img/HighContrastImages/HC Reference.tif";
const std::string image2name = "img/HighContrastImages/HCY00 X05.tif";
static const int IMAGE_WIDTH_ = 487;
static const int IMAGE_HEIGHT_ = 325;
#endif

#if 0
const std::string image1name = "img/big1.tif";
const std::string image2name = "img/big2.tif";
static const int IMAGE_WIDTH_ = 4870;
static const int IMAGE_HEIGHT_ = 3250;
#endif

#if 0
const std::string image1name = "img/1000_1.tif";
const std::string image2name = "img/1000_2.tif";
static const int IMAGE_WIDTH_ = 1000;
static const int IMAGE_HEIGHT_ = 1000;
#endif


#if 1
const std::string image1name = "img/2500_1.tif";
const std::string image2name = "img/2500_1_wave.tif";
static const int IMAGE_WIDTH_ = 2500;
static const int IMAGE_HEIGHT_ = 2500;
#endif

#if 0
const std::string image1name = "img/test/ref1.tif";
const std::string image2name = "img/test/skrecenie1.tif";
static const int IMAGE_WIDTH_ = 1500;
static const int IMAGE_HEIGHT_ = 1500;
#endif


#if 0
const std::string image1name = "img/samples1/oht_cfrp_00.tiff";
const std::string image2name = "img/samples1/oht_cfrp_02.tiff";
static const int IMAGE_WIDTH_ = 400;
static const int IMAGE_HEIGHT_ = 1040;
#endif

#if 0
const std::string image1name = "img/dziura1_.tif";
const std::string image2name = "img/dziura2_.tif";
static const int IMAGE_WIDTH_ = 2250;
static const int IMAGE_HEIGHT_ = 1640;
#endif


#if 0
const std::string image1name = "img/ref.tif";
//const std::string image2name = "img/ref_manual.tif";
//const std::string image2name = "img/deform.tif";
const std::string image2name = "img/disp_y.tif";
static const int IMAGE_WIDTH_ = 1000;
static const int IMAGE_HEIGHT_ = 1000;
#endif

#if 0
const std::string image1name = "img/big1_.tif";
const std::string image2name = "img/big1_disp_x.tif";
//const std::string image2name = "img/big1_.tif";
static const int IMAGE_WIDTH_ = 500;
static const int IMAGE_HEIGHT_ = 500;
#endif

#if 0
const std::string image1name = "img/tiff/trxy_s2_00.tif";
const std::string image2name = "img/tiff/trxy_s2_01.tif";
static const int IMAGE_WIDTH_ = 512;
static const int IMAGE_HEIGHT_ = 512;
#endif


static const int PATCH_SIZE = 15;// rozwiazac ten problem za posrednictwem parametrow wejsciowych
static const int SUB_IMG = 19;
static const int RESIZE_COEFF = 1;

static const int IMAGE_WIDTH = IMAGE_WIDTH_*RESIZE_COEFF;
static const int IMAGE_HEIGHT = IMAGE_HEIGHT_*RESIZE_COEFF;
//static const int PATCH_SIZE = PATCH_SIZE_*RESIZE_COEFF;// rozwiazac ten problem za posrednictwem parametrow wejsciowych
//static const int SUB_IMG = SUB_IMG_*RESIZE_COEFF;


//static const int IMAGE_WIDTH = 1366;
//static const int IMAGE_HEIGHT = 768;

//const string image1name = "img/img1.jpg";
//const string image2name = "img/img2.jpg";
#endif