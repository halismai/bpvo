#include "bpvo/types.h"
#include "bpvo/imwarp.h"

using namespace bpvo;

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

  Matrix33 K; K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;
  Matrix44 P(Matrix44::Identity());

  P.block<3,4>(0,0) = K * P.block<3,4>(0,0);


  alignas(16) float x[4*2];
  int n = warpPoints(P.data(), X, 4, x);
  printf("%d\n", n);

  for(int i = 0; i < static_cast<int>(sizeof(x)/sizeof(x[0])); ++i)
    printf("%f\n", x[i]);
  printf("\n");


  alignas(16) float c[4*4];
  n = computeInterpCoeffs(x, 4, c);
  for(int i = 0; i < 4; ++i) {
    printf("%f %f %f %f\n", c[4*i + 0], c[4*i + 1], c[4*i + 2], c[4*i + 3]);
  }


  __m128i ub = _mm_setr_epi32(640-2, 480-2, 640-2, 480-2);
  __m128i lb = _mm_setr_epi32(0, 0, 0, 0);

  __m128i p = _mm_setr_epi32(1, 20, -2, 85);
  __m128i m = _mm_and_si128(_mm_cmplt_epi32(p, ub),_mm_cmpgt_epi32(p, lb));

  int mask = _mm_movemask_epi8(m);
  int v1 = (mask & 0x00ff) == 0x00ff;
  int v2 = (mask & 0xff00) == 0xff00;
  printf("%d %d\n", v1, v2);

   return 0;
}
