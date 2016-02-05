#include <bpvo/linear_system_builder.h>
#include <bpvo/timer.h>
#include <iostream>

using namespace bpvo;

int main()
{
  size_t N = 10000;
  typename LinearSystemBuilder::JacobianVector  J(N);
  typename LinearSystemBuilder::ResidualsVector R(N);
  typename LinearSystemBuilder::ResidualsVector W(N);
  typename LinearSystemBuilder::ValidVector     V(N);

  typename LinearSystemBuilder::Hessian H;
  typename LinearSystemBuilder::Gradient G;

  for(size_t i = 0; i < N; ++i) {
    J[i].setRandom();
    R[i] = rand() / (float) RAND_MAX;
    W[i] = rand() / (float) RAND_MAX;
    V[i] = true; //(rand() / (float) RAND_MAX) > 0.1f;
  }

  float res_norm = LinearSystemBuilder::Run(J, R, W, V, &H, &G);

  std::cout << "res_norm: " << res_norm << "\n";
  std::cout << "res_norm: " << LinearSystemBuilder::Run(J,R,W,V) << std::endl;

  std::cout << "H\n" << H << std::endl;
  std::cout << "G\n" << G << std::endl;

  return 0;
}

