#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <immintrin.h>
#include "bpvo/imgproc.h"

#include <opencv2/highgui/highgui.hpp>

static void print(const char* s, const __m128& v)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, v);
  printf("%s: %g %g %g %g\n", s, buf[0], buf[1], buf[2], buf[3]);
}

struct IsLocalMaxSimd
{
  inline IsLocalMaxSimd() {}
  inline IsLocalMaxSimd(const float* ptr, int stride, int radius)
      : _ptr(ptr), _stride(stride), _radius(radius) {}

  bool operator()(int row, int col) const
  {
    const float* p = _ptr + row*_stride + col;
    auto v = _mm_set1_ps(*p);

    return
        13 == _mm_movemask_ps(_mm_cmpgt_ps(v, _mm_loadu_ps(p - 1       ))) &&
        15 == _mm_movemask_ps(_mm_cmpgt_ps(v, _mm_loadu_ps(p - 1 - _stride))) &&
        15 == _mm_movemask_ps(_mm_cmpgt_ps(v, _mm_loadu_ps(p - 1 + _stride)));
  }

  const float* _ptr;
  int _stride;
  int _radius;
}; // IsLocalMaxSimd

int main()
{
  int rows = 5, cols = 5;
  std::vector<float> v(rows * cols, 0);
  for(int i = 0; i < rows*cols; ++i) {
    v[i] = uint8_t( 255.0 * ( rand() / (float) RAND_MAX ) );
  }


  int y = 1, x = 1;
  float* ptr = v.data() + y*cols + x;
  *ptr = 5;

  auto p_1 = _mm_loadu_ps( ptr - 1 - cols );
  auto p_0 = _mm_loadu_ps( ptr - 1 );
  auto p_2 = _mm_loadu_ps( ptr + 1 );
  auto val = _mm_set1_ps( *ptr );
  //auto mask = _mm_cvtepi32_ps(_mm_setr_epi32(0, 0xffffffff, 0, 0));
  auto mask = _mm_setr_ps(0, 0xffffffff, 0, 0);
  print("mask", mask);

  auto m1 = _mm_cmpgt_ps(val, p_1);
  auto m2 = _mm_cmpgt_ps(val, p_0);
  auto m3 = _mm_cmpgt_ps(val, p_2);
  print("m1", m1);
  print("m2", m2);
  print("m3", m3);

  int v1 = _mm_movemask_ps(m1),
      v2 = _mm_movemask_ps(m2),
      v3 = _mm_movemask_ps(m3);
  printf("%d %d %d\n", v1, v2, v3);
  if(v1 == 15 && v2 == 13 && v3 == 15)
    printf("local max\n");
  else
    printf("not local max\n");

  cv::Mat I = imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  I.convertTo(I, CV_32F);

  {
    bpvo::IsLocalMax<float> is_local_max(I.ptr<float>(), I.cols, 1);
    cv::Mat D(I.size(), CV_8U, cv::Scalar(0));
    for(int y = 5; y < I.rows - 6; ++y)
    {
      for(int x = 5; x < I.cols - 6; ++x)
      {
        D.at<uint8_t>(y,x) = 255 * is_local_max(y, x);
      }
    }
    cv::imwrite("local_max.pgm", D);
  }

  {
    IsLocalMaxSimd is_local_max(I.ptr<float>(), I.cols, 1);
    cv::Mat D(I.size(), CV_8U, cv::Scalar(0));
    for(int y = 5; y < I.rows - 6; ++y)
    {
      for(int x = 5; x < I.cols - 6; ++x)
      {
        D.at<uint8_t>(y,x) = 255 * is_local_max(y, x);
      }
    }
    cv::imwrite("local_max_simd.pgm", D);
  }

}

