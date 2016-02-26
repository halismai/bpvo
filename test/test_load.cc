#include <immintrin.h>
#include <vector>
#include <cstdio>
#include <cstdint>

static void print(const char* s, const __m128& v)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, v);
  printf("%s: %g %g %g %g\n", s, buf[0], buf[1], buf[2], buf[3]);
}

int main()
{
  int rows = 30, cols = 40;
  std::vector<float> v(rows*cols);
  float* p = v.data();
  for(int i = 0; i < rows*cols; ++i) {
    p[i] = uint8_t( 255.0 * (rand()/(float)RAND_MAX) );
  }

  int x = 10, y = 13;
  int i = y*cols + x;

  auto a = _mm_loadu_ps( p + i ),
       b = _mm_loadu_ps( p + i + cols );

  print("a", a);
  print("b", b);
  print("", _mm_shuffle_ps(a, b, _MM_SHUFFLE(1,0,1,0)));

  printf("should have %g %g %g %g\n", *(p + i), *(p + i + 1), *(p + i + cols), *(p + i + cols + 1));

  return 0;
}
