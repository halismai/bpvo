#include <bpvo/types.h>
#include <iostream>

using namespace bpvo;
using namespace Eigen;

void projectPoint(const Matrix34& P, const Vector4f& X)
{
  Vector3f x = P * X; x /= x[2];

  // interpolation stuff

  float xf = x[0] - std::floor(x[0]);
  float yf = x[1] - std::floor(x[1]);

  printf("proj: %f %f\n", x[0], x[1]);
  printf("    : %g %g %g %g\n", xf, yf, 1.0-xf, 1-yf);
  printf("WWWW: %g %g %g %g\n", (1-xf)*(1-yf), xf*(1-yf), (1-xf)*yf, xf*yf);
  printf("\n");
}

std::ostream& operator<<(std::ostream& os, const __m128& v)
{
  Vector4f d;
  _mm_store_ps(d.data(), v);
  os << d.transpose();
  return os;
}

struct SseRoundTowardZero
{
  SseRoundTowardZero() : _mode(_MM_GET_ROUNDING_MODE()) {
    if(_mode != _MM_ROUND_TOWARD_ZERO)
      _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
  }

  ~SseRoundTowardZero() {
    if(_mode != _MM_ROUND_TOWARD_ZERO)
      _MM_SET_ROUNDING_MODE(_mode);
  }

  int _mode;
}; // SseRoundTowardZero

__m128 makeWeights(const __m128& p)
{
  auto w0 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 1, 0, 1)),
       w1 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 3, 3));

  std::cout << "have: " << p << std::endl;
  std::cout << "got : " << w0 << std::endl;
  std::cout << "got : " << w1 << std::endl;

  return _mm_mul_ps(w0, w1);
}

void projectTwoPointsSse(const Matrix34& P, const Vector4f& X0_, const Vector4f& X1_)
{
  SseRoundTowardZero round_to_zero;

  const auto P0 = _mm_setr_ps(P(0,0), P(0,1), P(0,2), P(0,3));
  const auto P1 = _mm_setr_ps(P(1,0), P(1,1), P(1,2), P(1,3));
  const auto P2 = _mm_setr_ps(P(2,0), P(2,1), P(2,2), P(2,3));
  const auto X0 = _mm_load_ps(X0_.data());
  const auto X1 = _mm_load_ps(X1_.data());

  auto x0 = _mm_mul_ps(P0, X0), x1 = _mm_mul_ps(P0, X1),
       y0 = _mm_mul_ps(P1, X0), y1 = _mm_mul_ps(P1, X1),
       z0 = _mm_mul_ps(P2, X0), z1 = _mm_mul_ps(P2, X1),
       xy = _mm_hadd_ps( _mm_hadd_ps(x0,y0), _mm_hadd_ps(x1,y1) ),
       zz = _mm_hadd_ps( _mm_hadd_ps(z0,z0), _mm_hadd_ps(z1,z1) ),
       p0p1 = _mm_div_ps(xy, zz);

  auto p_i = _mm_cvtepi32_ps(_mm_cvtps_epi32(p0p1));
  auto p_f = _mm_sub_ps(p0p1, p_i),
       p_f_1 = _mm_sub_ps(_mm_set1_ps(1.0), p_f);


  auto w0 = makeWeights( _mm_unpacklo_ps(p_f, p_f_1) );
  std::cout << "weight0: " << w0 << std::endl;
  auto w1 = makeWeights( _mm_unpackhi_ps(p_f, p_f_1) );
  std::cout << "weight1: " << w1 << std::endl;
}

void TestMask()
{
  int cols = 640, rows = 480;

  alignas(16) int data[4] = {cols, -1, 3, -4};
  const auto LB = _mm_setr_epi32(-1, -1, -1, -1);
  const auto UB = _mm_setr_epi32(cols-1, rows-1, cols-1, rows-1);

  auto v = _mm_load_si128((const __m128i*) data);
  int m = _mm_movemask_epi8(_mm_and_si128(_mm_cmpgt_epi32(v, LB),
                        _mm_cmplt_epi32(v, UB)));
  // 0xffff
  // 0xff00 // first point
  // 0x00ff // second point

  printf("1-st ok = %d\n", 0x00ff == (m & 0x00ff));
  printf("2-nd ok = %d\n", 0xff00 == (m & 0xff00));
}

int main()
{
  Matrix34 P;
  P << 600.0, 0.0, 320.0, 0.0,
       0.0, 600.0, 240.0, 0.0,
       0.0, 0.0, 1.0, 0.0;

  Vector4f X0(1.2, 2.3, 3.4, 1.0);
  Vector4f X1(4.31, 5.8, 6.01, 1.0);

  projectPoint(P, X0);
  projectTwoPointsSse(P, X0, X1);

  projectPoint(P, X1);

  TestMask();
}

