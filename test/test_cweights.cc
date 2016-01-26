#include <pmmintrin.h>
#include <cstdio>
#include <iostream>

#include <bpvo/types.h>

void print(const char* s, const __m128& v)
{
  alignas(16) float data[4];
  _mm_store_ps(data, v);
  printf("%s %f %f %f %f\n", s, data[0], data[1], data[2], data[3]);
}

using namespace bpvo;

Eigen::Vector2f proj(const Matrix34& P, const Eigen::Vector4f& X)
{
  Eigen::Vector3f x = P * X;
  return Eigen::Vector2f(x[0] / x[2], x[1] / x[2]);
}

Eigen::Vector4f interpWeights(const Eigen::Vector2f& x)
{
  float xf = x[0] - floor(x[0]),
        yf = x[1] - floor(x[1]);

  return Eigen::Vector4f(
      (1.0 - yf) * (1.0 - xf),
      (1.0 - yf) * xf,
      yf * (1.0 - xf),
      yf * xf);
}

void TestShuffle()
{
  alignas(16) float data[4] = {1.0, 2.0, 3.0, 4.0};
  auto p = _mm_load_ps(data);
  // [1   2  3   4]
  // [x,  1-x, y, 1-y]
  // [2, 1, 2, 1]
  // [x-1, x, 1-x, x]
  // and
  //  [4, 4, 3, 3]
  // [1-y, 1-y, y, y]
  //

  auto a1 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,1,0,1)),
       a2 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,3,3));

  print("a1:", a1);
  print("a2:", a2);
}

__m128 makeWeights(const __m128& p)
{
  /*
  auto a1 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,1,0,1)),
       a2 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,3,3));
       */

  auto a1 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,0,1,0)),
       a2 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3,3,2,2));

  return _mm_mul_ps(a1, a2);
}


int main()
{
  TestShuffle();

  Matrix34 P;
  P << 600.0, 0.0, 320.0, 0.0,
       0.0, 600.0, 240.0, 0.0,
       0.0, 0.0, 1.0, 0.0;

  Eigen::Vector4f pt0, pt1;
  pt0 << 1.5, 2.1, 3.2, 1.0;
  pt1 << 4.1, 5.2, 6.3, 1.0;

  int cols = 640, rows = 480;
  const auto P0 = _mm_setr_ps(P(0,0), P(0,1), P(0,2), P(0,3));
  const auto P1 = _mm_setr_ps(P(1,0), P(1,1), P(1,2), P(1,3));
  const auto P2 = _mm_setr_ps(P(2,0), P(2,1), P(2,2), P(2,3));
  const auto UB = _mm_setr_ps(cols-2, rows-2, cols-2, rows-2);
  const auto ZERO = _mm_setzero_ps();
  const auto ONE  = _mm_set1_ps(1.0f);

  auto X0 = _mm_load_ps(pt0.data()),
       X1 = _mm_load_ps(pt1.data());

  print("X0: ", X0);

  auto x0 = _mm_mul_ps(P0, X0), x1 = _mm_mul_ps(P0, X1);
  auto y0 = _mm_mul_ps(P1, X0), y1 = _mm_mul_ps(P1, X1);
  auto z0 = _mm_mul_ps(P2, X0), z1 = _mm_mul_ps(P2, X1);
  auto xy = _mm_hadd_ps(_mm_hadd_ps(x0,y0), _mm_hadd_ps(x1,y1));
  auto zz = _mm_hadd_ps(_mm_hadd_ps(z0,z0), _mm_hadd_ps(z1,z1));

  // projections for the two points
  auto x0y0x1y1 = _mm_div_ps(xy, zz);
  auto x0y0x1y1_i = _mm_cvtps_epi32(x0y0x1y1); // the floor
  // fractional part
  auto x0y0x1y1_f = _mm_sub_ps(x0y0x1y1, _mm_cvtepi32_ps(x0y0x1y1_i));
  // 1 minus
  auto x0y0x1y1_1_f = _mm_sub_ps(ONE, x0y0x1y1_f);

  auto W0 = makeWeights(_mm_unpacklo_ps(x0y0x1y1_f, x0y0x1y1_1_f));
  auto W1 = makeWeights(_mm_unpackhi_ps(x0y0x1y1_f, x0y0x1y1_1_f));

  auto pp0 = _mm_unpacklo_ps(x0y0x1y1_f, x0y0x1y1_1_f);
  print("x0y0x1y1    :   ", x0y0x1y1);

  print("x0y0x1y1_f  : ", x0y0x1y1_f);
  print("x0y0x1y1_1_f: ", x0y0x1y1_1_f);
  print("pp0         : ", pp0);

  {
    auto p0 = proj(P, pt0);
    auto w0 = interpWeights(p0);
    std::cout << p0.transpose() << "\n";
    std::cout << "w0:" << w0.transpose() << "\n";
  }
  print("W0: ", W0);


}
