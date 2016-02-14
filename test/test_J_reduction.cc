#include <immintrin.h>
#include <stdio.h>
#include <string.h>

void print(__m128 v)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, v);
  printf("%f %f %f %f", buf[0], buf[1], buf[2], buf[3]);
}

int main()
{
  alignas(16) float data[24] = {0.0f};
  alignas(16) float J[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0};

  float w = 0.5f;
  float r = 2.1f;

  __m128 wwww = _mm_set1_ps(w);
  __m128 v1234 = _mm_loadu_ps(J);
  __m128 v56xx = _mm_loadu_ps(J + 4);

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v1212, v1212));

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  for(int i = 0; i < 12; ++i)
    printf("%f ", data[i]);

  printf("\n");

  __m128 v3344 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v5656, v5656));
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));


}
