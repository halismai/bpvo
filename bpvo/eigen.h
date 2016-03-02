#ifndef BPVO_EIGEN_H
#define BPVO_EIGEN_H

#include <Eigen/Core>

namespace bpvo {

template <class Derived> inline
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime - 1, 1>
normHomog(const Eigen::MatrixBase<Derived>& p)
{
  static_assert(
      Derived::RowsAtCompileTime != Eigen::Dynamic &&
      Derived::ColsAtCompileTime == 1,
      "input must be a column vector with known at compile time");

  constexpr int R = Derived::RowsAtCompileTime;
  typedef typename Derived::Scalar T;

  return Eigen::Matrix<T, R-1, 1>( (T(1) / p[R-1]) * p.template head<R-1>());
}

template <class Derived> inline
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime + 1, 1>
formHomog(const Eigen::MatrixBase<Derived>& p)
{
  static_assert(
      Derived::RowsAtCompileTime != Eigen::Dynamic &&
      Derived::ColsAtCompileTime == 1,
      "input must be a column vector with known at compile time");

  constexpr int R = Derived::RowsAtCompileTime;
  typedef typename Derived::Scalar T;

  Eigen::Matrix<T,R+1,1> ret;
  ret.template head<R>() = p;
  ret[R] = T(1);

  return ret;
}

template <typename T, int M, int N> using Mat_ = Eigen::Matrix<T,M,N>;
template <typename T, int M> using Vec_ = Mat_<T,M,1>;
template <typename T, int N> using RowVec_ = Mat_<T,1,N>;

}; // bpvo

#endif // BPVO_EIGEN_H
