#include <bpvo/linear_system_builder.h>
#include <bpvo/timer.h>
#include <iostream>

using namespace bpvo;

void doLinearSystemReduction(const typename LinearSystemBuilder::JacobianVector& J,
                             const typename LinearSystemBuilder::ResidualsVector& R,
                             const typename LinearSystemBuilder::ValidVector& V,
                             typename LinearSystemBuilder::Hessian& H,
                             typename LinearSystemBuilder::Gradient& G)
{
  H.setZero();
  G.setZero();

  for(size_t i = 0; i < J.size(); ++i) {
    if(V[i]) {
      H.noalias() += J[i].transpose() * J[i];
      G.noalias() += J[i].transpose() * R[i];
    }
  }
}

int main()
{
  int N = 20 * 1000; //480 * 640 * 8;

  typename LinearSystemBuilder::JacobianVector J(N);
  typename LinearSystemBuilder::ResidualsVector R(N);
  typename LinearSystemBuilder::ValidVector V(N);

  for(int i = 0; i < N; ++i) {
    J[i].setRandom();
    R[i] = (rand() / (double) RAND_MAX);
    V[i] = (rand() / (double) RAND_MAX) > 0.1;
  }

  typename LinearSystemBuilder::Hessian A;
  typename LinearSystemBuilder::Gradient b;

  LinearSystemBuilder sys_builder(LossFunctionType::kHuber);
  float r_norm = sys_builder.run(J, R, V, A, b);
  std::cout << "GOT:  " << r_norm << std::endl;

  decltype(A) A2;
  decltype(b) b2;

  A2.setZero();
  b2.setZero();

  float r_norm2 = 0.0;
  for(int i = 0; i < N; ++i) {
    if(V[i]) {
      A2.noalias() += J[i].transpose() * J[i];
      b2.noalias() += J[i].transpose() * R[i];
      r_norm2 += R[i]*R[i];
    }
  }
  r_norm2 = std::sqrt(r_norm2);

  std::cout << "\n" << A << "\n" << std::endl;
  std::cout << "\n" << A2 << "\n" << std::endl;

  /*
  std::cout << (A2 - A) << std::endl;
  std::cout << std::endl;
  std::cout << (b2 - b) << std::endl;
  */
  std::cout << "GOT: " << r_norm2 << std::endl;


  printf("\n\nTiminng for %d points\n", N);

  auto t = TimeCode(50, [&] () { sys_builder.run(J, R, V, A, b); });
  printf("time: %0.2f ms\n", t);


  std::cout << "ERROR:\n" << (A2-A) << std::endl;
  std::cout << "ERROR:\n" << (b2-b) << std::endl;

  return 0;
}
