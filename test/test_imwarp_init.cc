#include "bpvo/types.h"
#include "bpvo/interp_util.h"
#include <immintrin.h>
#include <iostream>
#include <vector>

using namespace bpvo;

std::vector<float> interp_init_basic(const float* X)
{
  Matrix33 K;
  K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;

  std::vector<float> ret;

  for(int i = 0; i < 4; ++i)
  {
    Eigen::Vector3f x = K * Point::Map(X + 4*i).head<3>();
    x /= x[2];

    float xf = x[0], yf = x[1];
    int xi = static_cast<int>(xf), yi = static_cast<int>(yf);

    xf -= xi;
    yf -= yi;

    float xfyf = xf*yf;
    ret.push_back( xfyf - yf - xf + 1.0f );
    ret.push_back( xf - xfyf );
    ret.push_back( yf - xfyf );
    ret.push_back( xfyf );
  }

  return ret;
}

void interp_sse(const float* X, float *c)
{
  Matrix33 K;
  K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;

  Eigen::Vector3f x0, x1;

  static const __m128 ONES = _mm_set1_ps(1.0f);

  __m128 xf0, xf1, xi0, xi1, wx0, wx1, xx0, xx1, yy0, yy1;

  int i = 0;
  x0 = K * Point::Map(X + 4*i + 0).head<3>();
  x0 /= x0[2];

  x1 = K * Point::Map(X + 4*i + 4).head<3>();
  x1 /= x1[2];

  xf0 = _mm_setr_ps(x0[0], x0[1], x1[0], x1[1]);

  xi0 = _mm_floor_ps(xf0);
  xf0 = _mm_sub_ps(xf0, xi0);

  wx0 = _mm_sub_ps(ONES, xf0);
  xx0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(0,0,0,0));
  xx0 = _mm_shuffle_ps(xx0, xx0, _MM_SHUFFLE(2,0,2,0));
  yy0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(1,1,1,1));
  _mm_store_ps(c + 4*i + 0, _mm_mul_ps(xx0, yy0));

  xx0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(2,2,2,2));
  xx0 = _mm_shuffle_ps(xx0, xx0, _MM_SHUFFLE(2,0,2,0));
  yy0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(3,3,3,3));
  _mm_store_ps(c + 4*i + 4, _mm_mul_ps(xx0, yy0));


  x0 = K * Point::Map(X + 4*i + 8).head<3>();
  x0 /= x0[2];

  x1 = K * Point::Map(X + 4*i + 12).head<3>();
  x1 /= x1[2];

  xf1 = _mm_setr_ps(x0[0], x0[1], x1[0], x1[1]);

  xi1 = _mm_floor_ps(xf1);
  xf1 = _mm_sub_ps(xf1, xi1);

  wx1 = _mm_sub_ps(ONES, xf1);
  xx1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(0,0,0,0));
  xx1 = _mm_shuffle_ps(xx1, xx1, _MM_SHUFFLE(2,0,2,0));
  yy1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(1,1,1,1));
  _mm_store_ps(c + 4*i + 8, _mm_mul_ps(xx1, yy1));

  xx1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(2,2,2,2));
  xx1 = _mm_shuffle_ps(xx1, xx1, _MM_SHUFFLE(2,0,2,0));
  yy1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(3,3,3,3));
  _mm_store_ps(c + 4*i + 12, _mm_mul_ps(xx1, yy1));
}


int main()
{
  alignas(16) float X [] =
  {
    -18.819, -18.2095,  46.8571,        1,
    -18.2857, -18.2095,  46.8571,        1,
    -17.7524, -18.2095,  46.8571,        1,
    -16.6857, -18.2095,  46.8571,        1,
    -16.1524, -18.2095,  46.8571,        1
  }; // X


  auto c1 = interp_init_basic(X);

  alignas(16) float c2[4*4];
  memset(c2, 0, sizeof(c2));
  interp_sse(X, c2);

  for(int i = 0; i < 4; ++i) {
    std::cout << "c1: " << Point::Map(c1.data() + 4*i).transpose() << std::endl;
    std::cout << "c2: " << Point::Map(c2 + 4*i).transpose() << std::endl;
    std::cout << "\n";
  }

  return 0;
}
