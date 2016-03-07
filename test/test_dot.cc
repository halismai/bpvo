#include <immintrin.h>
#include <Eigen/Core>
#include <iostream>
#include <cstdio>
#include <vector>

#include "bpvo/timer.h"

static inline void printm128(const char* prefix, const __m128& x)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, x);
  printf("%s: [%f %f %f %f]\n", prefix, buf[0], buf[1], buf[2], buf[3]);
}

static inline float DOT(const float* a, const float* b)
{
  float ret;
  _mm_store_ss(&ret, _mm_dp_ps(_mm_load_ps(a), _mm_load_ps(b), 0xff));
  return ret;
}


int main()
{
  Eigen::Vector4f a(1.0, 2.0, 3.0, 4.0);
  Eigen::Vector4f b(5.0, 6.0, 7.0, 8.0);

  std::cout << a.dot(b) << std::endl;

  auto d = _mm_dp_ps(_mm_load_ps(a.data()), _mm_load_ps(b.data()), 0xff);
  printm128("d: ", d);


  {
    std::vector<float> vals(10);
    auto code = [&]()
    {
      for(int i = 0; i < 100; ++i)
        vals[i % vals.size()]  = a.dot(b);
    };

    auto t = bpvo::TimeCode(10000000, code);
    printf("Eigen time %f\n", t);
  }

  {
    std::vector<float> vals(10);
    auto code = [&]()
    {
      for(int i = 0; i < 100; ++i)
        vals[i % vals.size()]  = DOT(a.data(), b.data());
    };

    auto t = bpvo::TimeCode(10000000, code);
    printf("SSE4  time %f\n", t);
  }

  printf("%f %f\n", a.dot(b), DOT(a.data(), b.data()));

  return 0;
}
