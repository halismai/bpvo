#include <bpvo/types.h>
#include <iostream>

using namespace bpvo;

template <class Derived> static inline
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime - 1, 1>
normHomog(const Eigen::MatrixBase<Derived>& p)
{
  auto w_i = 1.0f / p[Derived::RowsAtCompileTime-1];
  return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime-1,1>(
      w_i * p.template head<Derived::RowsAtCompileTime-1>());
}


int main()
{

  Eigen::Matrix<float,3,1> X(1,2,3);
  std::cout << X.transpose() << std::endl;
  std::cout << normHomog(X).transpose() << std::endl;

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
