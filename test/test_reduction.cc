#include "bpvo/rigid_body_warp.h"
#include "bpvo/timer.h"
#include "bpvo/vector6.h"
#include <iostream>

using namespace bpvo;

static inline Eigen::Matrix<float,6,6> MakeFull(const float* data)
{
  Eigen::Matrix<float,6,6> ret;
  for(int i = 0, ii=0; i < 6; i += 2) {
    for(int j = i; j < 6; j += 2) {
      ret(i  , j  ) = data[ii++];
      ret(i  , j+1) = data[ii++];
      ret(i+1, j  ) = data[ii++];
      ret(i+1, j+1) = data[ii++];
    }
  }

  ret.selfadjointView<Eigen::Upper>().evalTo(ret);
  return ret;
}

static inline void RankUpdateEigen(const Eigen::Matrix<float,1,6>& J, float w,
                                   float* data)
{
  __m128 wwww = _mm_set1_ps(w);
  __m128 v1234 = _mm_loadu_ps(J.data());
  __m128 v56xx = _mm_loadu_ps(J.data() + 4);

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v1212, v1212));

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  __m128 v3344 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v5656, v5656));
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));
}

int main()
{
  int N = 640*480;

  typedef Eigen::Matrix<float,1,6> RowVector6;
  typename EigenAlignedContainer<RowVector6>::type J1(N);
  for(int i = 0; i < N; ++i)
    J1[i].setRandom();

  typename EigenAlignedContainer<Vector6>::type J2(N);
  for(int i = 0; i < N; ++i)
    J2[i] = Vector6(J1[i].data());


  typedef Eigen::Matrix<float,6,6> Hessian;
  Hessian H1(Hessian::Zero());
  Hessian H2(Hessian::Zero());
  Hessian H3(Hessian::Zero());

  {
    auto t = TimeCode(100, [&]() {
                      for(size_t i = 0; i < J1.size(); ++i)
                        H1.noalias() += J1[i].transpose() * J1[i];
                      });
    printf("Eigen time %f\n", t);
  }

  {
    alignas(16) float buf[24];
    memset(buf, 0, sizeof(buf));

    auto t = TimeCode(100, [&]() {
                      for(size_t i = 0; i < J2.size(); ++i)
                        Vector6::RankUpdate(J2[i], 1.0, buf);
                      });
    printf("vector6 time %f\n", t);
    H2 = MakeFull(buf);
  }

  {
    alignas(16) float buf[24];
    memset(buf, 0, sizeof(buf));

    auto t = TimeCode(100, [&]() {
                      for(size_t i = 0; i < J2.size(); ++i)
                        RankUpdateEigen(J1[i], 1.0, buf);
                      });
    printf("unaligned time %f\n", t);
    H3 = MakeFull(buf);
  }


  std::cout << (H1-H2) << std::endl;
  return 0;
}
