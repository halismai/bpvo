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
  Hessian H2;

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

  std::cout << (H1-H2) << std::endl;
  return 0;
}
