#ifndef PHOTOBUNDLE_EIGEN_H
#define PHOTOBUNDLE_EIGEN_H

#include "types.h"


/**
 * Normalizes a vector by deviding with the last coordinate
 */
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

/**
 * makes a vector homogenous by append '1' as the last coordinate
 */
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

#endif // PHOTOBUNDLE_TYPES_H

