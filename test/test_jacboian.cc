#include "bpvo/rigid_body_warp.h"
#include "bpvo/timer.h"

using namespace bpvo;

typedef RigidBodyWarp Warp;
typename Warp::PointVector GetPoints(int N = 640*480)
{
  N = N - (N % 16);

  typename Warp::PointVector points(N);
  for(int i = 0; i < N; ++i) {
    points[i].setRandom();
    points[i][3] = 1.0;
  }

  return points;
}

int main()
{
  Matrix33 K;
  K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;
  RigidBodyWarp warp(K, 0.1);

  auto points = GetPoints();
  std::vector<float> IxIy(points.size()*2);
  for(size_t i = 0; i < IxIy.size(); ++i)
    IxIy[i] = rand() / (float) RAND_MAX;


  typename Warp::JacobianVector J(points.size());

  {
    auto J_ptr = J[0].data();
    auto t = TimeCode(100, [&]() { warp.computeJacobian(points, IxIy.data(), J_ptr); });
    printf("simd time %f\n", t);
  }

  {
    auto t = TimeCode(100, [&]() {
                        for(size_t i = 0; i < points.size(); ++i)
                        {
                          warp.jacobian(points[i], IxIy[2*i+0], IxIy[2*i+1], J[i].data());
                        }
                      });
    printf("no simd time %f\n", t);
  }

  return 0;
}

