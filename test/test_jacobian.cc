#include "bpvo/rigid_body_warp.h"
#include "bpvo/timer.h"

using namespace bpvo;

typename RigidBodyWarp::PointVector MakePoints(int N = 1 << 14)
{
  typename RigidBodyWarp::PointVector ret(N);
  for(int i = 0; i < N; ++i) {
    ret[i].setRandom();
    ret[i][3] = 1.0;
  }

  return ret;
}

template <class Warp> typename Warp::
JacobianVector computeJacobians(const Warp& warp, typename Warp::PointVector& X)
{
  typename Warp::JacobianVector ret(X.size());
  for(size_t i = 0; i < X.size(); ++i)
    warp.jacobian(X[i], 1, 2, ret[i].data());
  return ret;
}

template <class Warp> typename Warp::
JacobianVector computeJacobians2(const Warp& warp, typename Warp::PointVector& X)
{
  typename Warp::JacobianVector ret(X.size());
  Eigen::Matrix<float, Eigen::Dynamic, 6> J(X.size(), 6);

  __m128 FX = _mm_set1_ps(615.0f);
  __m128 FY = _mm_set1_ps(615.0f);
  __m128 C1 = _mm_set1_ps(0.0f);
  __m128 C2 = _mm_set1_ps(0.0f);
  __m128 C3 = _mm_set1_ps(0.0f);
  __m128 S = _mm_set1_ps(1.0f);
  __m128 Ix = _mm_set1_ps(1.0f);
  __m128 Iy = _mm_set1_ps(1.0f);

  int N = X.size();

  for(int j = 0; j < 6; ++j)
  {
    auto* ptr = &J(0,j);
    for(int i = 0; i < N; ++i)
    {
      auto xyzw = _mm_load_ps( X[i].data() );
      auto x = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(0,0,0,0));
      auto y = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(1,1,1,1));
      auto z = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(2,2,2,2));

      auto z2 = _mm_mul_ps(z, z);
      auto y2 = _mm_mul_ps(y, y);
      auto xy = _mm_mul_ps(x, y);

      auto jj = _mm_div_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(Iy, y2), _mm_mul_ps(Iy, z2)),
                                      _mm_mul_ps(Ix, xy)), z2);
      jj = _mm_xor_ps(jj, _mm_set1_ps(-0.0));
      _mm_store_ps(ptr + i, jj);
    }
  }


  Eigen::Matrix<float, 6, Eigen::Dynamic> Jt = J.transpose();
  memcpy(ret.data(), Jt.data(), 6*X.size()*sizeof(float));
  return ret;
}

int main()
{
  auto X = MakePoints(640*480);
  printf("%zu\n", X.size());

  typename RigidBodyWarp::JacobianVector J;
  Matrix33 K; K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;
  RigidBodyWarp warp(K, 0.1);

  {
    auto t = TimeCode(100, [&]() { J = computeJacobians(warp, X); });
    printf("Time %f\n", t);
    J.clear();
  }

  {
    auto t = TimeCode(100, [&]() { J = computeJacobians2(warp, X); });
    printf("Time %f\n", t);
    J.clear();
  }
}
