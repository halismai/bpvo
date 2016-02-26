#include <bpvo/types.h>
#include <iostream>

using namespace bpvo;

int main()
{
  typedef Eigen::Matrix<float,1,6> Jacobian;
  typedef typename EigenAlignedContainer<Jacobian>::type JacobianVector;

  Jacobian jj; jj << 1, 2, 3, 4, 5, 6;

  JacobianVector J;
  for(int i = 0; i < 2; ++i)
  {
    J.push_back( jj );
  }

  static constexpr int N = 2;
  Eigen::Matrix<float, N, 6> JM;
  JM.row(0) << 1, 2, 3, 4, 5, 6;
  JM.row(1) << 1, 2, 3, 4, 5, 6;

  std::cout << JM << std::endl;

  auto* p = JM.data();
  for(int i = 0; i < 2*6; ++i)
    printf("%f ", p[i]);
  printf("\n");

  printf("\n");
  std::cout << Eigen::Matrix<float,6,2>::Map(J[0].data()) << std::endl;


  Eigen::Matrix<float,6,N> Jt = JM.transpose();
  memcpy(J[0].data(), Jt.data(), N*6*sizeof(float));
  std::cout << "\n";
  for(const auto& jj : J)
    std::cout << jj << std::endl;

  return 0;
}
