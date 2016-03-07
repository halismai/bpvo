#include "bpvo/types.h"
#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

const int INTER_BITS = 5;
const int INTER_BITS2 = INTER_BITS * 2;
const int INTER_TAB_SIZE = 1 << INTER_BITS;
const int INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE;

const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

const int INTER_RESIZE_COEF_BITS = 11;
const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];


static void interpLinear(float x, float* coeffs)
{
  coeffs[0] = 1.0f - x;
  coeffs[1] = x;
}

static void initInterpTable1d(float* table, int table_size)
{
  float scale = 1.0f / table_size;
  for(int i = 0; i < table_size; ++i, table += 2)
    interpLinear(i*scale, table);
}

static void initInterpTable2d(float* tab, short* itab)
{
  constexpr int ks = 2;
  float _tab[ks*INTER_TAB_SIZE];
  initInterpTable1d(_tab, INTER_TAB_SIZE);

  for(int i = 0; i < INTER_TAB_SIZE; ++i)
  {
    for(int j = 0; j < INTER_TAB_SIZE; ++j, tab += ks*ks, itab += ks*ks)
    {
      int isum = 0;
      for(int k1 = 0; k1 < ks; ++k1)
      {
        float vy = _tab[i*ks + k1];
        for(int k2 = 0; k2 < ks; ++k2)
        {
          float v = vy*_tab[j*ks + k2];
          tab[k1*ks + k2] = v;
          itab[k1*ks + k2] = cv::saturate_cast<short>(v*INTER_REMAP_COEF_SCALE);
          isum += itab[k1*ks + k2];
        }
      }

      if(isum != INTER_REMAP_COEF_SCALE)
      {
        int diff = isum - INTER_RESIZE_COEF_SCALE;
        int ks2 = ks/2, Mk1=ks2, Mk2=ks2, mk1=ks2, mk2=ks2;
        for(int k1 = ks2; k1 < ks2+2; ++k1)
        {
          for(int k2 = k1; k2 < ks2+2; ++k2)
          {
            if(itab[k1*ks + k2] < itab[mk1*ks + mk2])
            {
              mk1 = k1;
              mk2 = k2;
            } else if(itab[k1*ks + ks2] > itab[Mk1*ks+Mk2])
            {
              Mk1 = k1;
              Mk2 = k2;
            }

            if(diff <0)
              itab[Mk1*ks + Mk2] = (short) (itab[Mk1*ks + Mk2] - diff);
            else
              itab[mk1*ks + mk2] = (short) (itab[mk1*ks + mk2] - diff);
          }
        }
      }
    }
  }

  tab -= INTER_TAB_SIZE*ks*ks;
  itab -= INTER_TAB_SIZE2*ks*ks;
}

int main()
{
  initInterpTable2d(BilinearTab_f[0][0], BilinearTab_i[0][0]);

  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  I.convertTo(I, CV_32FC1, 1.0, 0.0);
  THROW_ERROR_IF(I.empty(),  "could not read image");


  std::vector<float> x{2, 3, 1.25, -10};
  std::vector<float> y{5, 2, 3.2, 8};

  cv::Mat dst;
  dst.create(1, 4, CV_32FC1);
  cv::remap(I, dst, x, y, cv::INTER_LINEAR);//, cv::BORDER_CONSTANT, cv::Scalar(0));

  auto ptr = reinterpret_cast<const float*>(dst.data);
  printf("i: %g %g %g %g\n", ptr[0], ptr[1], ptr[2], ptr[3]);
  printf("n: %g %g\n", I.at<float>((int)y[0], (int)x[0]), I.at<float>((int)y[1], (int)x[1]));

  cv::Mat xy(1, 4, CV_32FC2);
  xy.at<cv::Point2f>(0,0) = cv::Point2f(1.0, 2.0);
  xy.at<cv::Point2f>(0,1) = cv::Point2f(3.0, 4.0);
  xy.at<cv::Point2f>(0,2) = cv::Point2f(5.0, 6.0);
  xy.at<cv::Point2f>(0,3) = cv::Point2f(7.0, 8.1);

  auto xy_ptr = xy.ptr<const float>();
  for(int i = 0; i < 4*2; ++i)
    printf("%g ", xy_ptr[i]);
  printf("\n");


  cv::Mat m1, m2;
  cv::convertMaps(xy, cv::Mat(), m1, m2, CV_16SC2);

  cv::Mat dst2;
  cv::remap(I, dst2, m1, m2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0));

  auto d_ptr = dst2.ptr<const float>();
  printf("i: %g %g %g %g\n", d_ptr[0], d_ptr[1], d_ptr[2], d_ptr[3]);
  printf("n: %g %g %g %g\n",
         I.at<float>( xy.at<cv::Point2f>(0,0) ),
         I.at<float>( xy.at<cv::Point2f>(0,1) ),
         I.at<float>( xy.at<cv::Point2f>(0,2) ),
         I.at<float>( xy.at<cv::Point2f>(0,3) ));

  return 0;
}

