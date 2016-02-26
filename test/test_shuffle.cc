#include <immintrin.h>
#include <cstdio>

static void print(const char* s, const __m128i& v)
{
  alignas(16) int buf[4];
  _mm_store_si128((__m128i*) buf, v);
  printf("%s: %d %d %d %d\n", s, buf[0], buf[1], buf[2], buf[3]);
}

static void print(const char* s, const __m128& v)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, v);
  printf("%s: %f %f %f %f\n", s, buf[0], buf[1], buf[2], buf[3]);
}

int main()
{
  // x0, y0, x1, y1
  __m128i a = _mm_setr_epi32(1, 2, 3, 4);
  __m128i b = _mm_setr_epi32(10, 20, 30, 40);

  print("a", a);
  print("b", b);

  // [1, 3, 10, 30]
  auto u0 = _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 2, 0));
  auto u1 = _mm_shuffle_epi32(b, _MM_SHUFFLE(0, 0, 2, 0));
  auto x = _mm_unpacklo_epi32(u0, u1);
  x = _mm_shuffle_epi32(x, _MM_SHUFFLE(3,1,2,0));
  print("x", x);

  // [2, 4, 20, 40]
  auto v0 = _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 1));
  auto v1 = _mm_shuffle_epi32(b, _MM_SHUFFLE(0, 0, 3, 1));
  auto y = _mm_unpacklo_epi32(v0, v1);
  print("y", y);
  y = _mm_shuffle_epi32(y, _MM_SHUFFLE(3,1,2,0));
  print("y", y);

  __m128 p = _mm_setr_ps(1, 2, 3, 4);
  print("p", p);
  print("x", _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0)));
  print("y", _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1)));
  print("z", _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2)));


  printf("\n\n");

  __m128 A[] = {
    _mm_setr_ps(1, 2, 3, 4),
    _mm_setr_ps(10, 20, 30, 40),
    _mm_setr_ps(100, 200, 300, 400),
    _mm_setr_ps(1000, 2000, 3000, 4000),
  }; // A

  {
    auto x1 = A[0],
         x2 = A[1],
         x3 = A[2],
         x4 = A[3];

    auto a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(0,0,0,0));
    auto b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(0,0,0,0));
    auto X = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

    print("x", X);

    a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(1,1,1,1));
    b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(1,1,1,1));
    print("Y", _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0)));

    a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
    b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
    print("Z", _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0)));
  }

  // [ix iy ix iy]
  __m128 G1 = _mm_setr_ps(1, 3, 2, 4);
  __m128 G2 = _mm_setr_ps(10, 30, 20, 40);

  // [1, 2, 10, 20]
  print("ix", _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(2,0,2,0)));
  // [3, 4, 30, 40]
  print("iy", _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(3,1,3,1)));


}
