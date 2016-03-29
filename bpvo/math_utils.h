/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#ifndef BPVO_MATH_UTILS_H
#define BPVO_MATH_UTILS_H

#include <bpvo/debug.h>
#include <bpvo/utils.h>

#include <Eigen/Core>
#include <Eigen/LU>

namespace bpvo {

namespace math {

FORCE_INLINE int Floor(double v)
{
#if defined(WITH_SIMD)
  auto t = _mm_set_sd(v);
  int i = _mm_cvtsd_si32(t);
  return i - _mm_movemask_pd(_mm_cmplt_sd(t, _mm_cvtsi32_sd(t,i)));
#else
  int i = static_cast<int>(v);
  return i - (i > v);
#endif
}

FORCE_INLINE int Floor(float v)
{
#if defined(WITH_SIMD)
  auto t = _mm_set_ss(v);
  int i = _mm_cvtss_si32(t);
  return i - _mm_movemask_ps(_mm_cmplt_ss(t, _mm_cvtsi32_ss(t,i)));
#else
  int i = (int) v;
  return i - (i > v);
#endif
}

FORCE_INLINE int Floor(int v)
{
  return v;
}

/**
 * Contanst PI
 */
template <typename T> FORCE_INLINE constexpr T pi()
{
  return assert_is_floating_point<T>(),
         static_cast<T>(3.14159265358979323846);
}

/**
 * converts an angle in degrees to radians
 */
template <typename T> FORCE_INLINE T deg2rad(const T d)
{
  assert_is_floating_point<T>();
  return (pi<T>()/180.0) * d;
}

/**
 * converts radians to degrees
 */
template <typename T> FORCE_INLINE T rad2deg(const T r)
{
  assert_is_floating_point<T>();
  return (180.0/pi<T>()) * r;
}

/**
 * squares the input number
 */
template <typename T> FORCE_INLINE T sq(const T v) { return v*v; }

/**
 * computes the inverse sqaure root of the input number
 */
template <typename T> FORCE_INLINE
T invsqrt(const T v) { return 1.0f / sqrt(v); }


/**
 * Makes a skew (anti-symmetric) 3x3 matrix from 3 numbers
 */
template <typename T = double> FORCE_INLINE
Eigen::Matrix<T, 3, 3> MakeSkew(T x, T y, T z)
{
  Eigen::Matrix<T, 3, 3> S;
  S << 0, -z, y,
       z, 0, -x,
       -y, x, 0;

  return S;
}


/**
 * Makes a skew (anti-symmetric) 3x3 matrix from an input 3-vector
 *
 * Generic version that should work with any Eigen type that derives from
 * MatrixBase
 */
template <class Derived> FORCE_INLINE
Eigen::Matrix<typename Derived::Scalar, 3, 3>
MakeSkew(const Eigen::MatrixBase<Derived>& v)
{
  Eigen::Matrix<typename Derived::Scalar, 3, 3> S;
  S <<    0 ,  -v[2],  v[1],
         v[2],    0 , -v[0],
        -v[1],  v[0],    0;

  return S;
}

/**
 * Converts a 6-vector (twsit) to a 4x4 homogenous rigid-body transformation
 * matrix
 */
template <typename Derived> FORCE_INLINE
Eigen::Matrix<typename Derived::Scalar, 4, 4>
TwistToMatrix(const Eigen::MatrixBase<Derived>& p)
{
  using namespace Eigen;
  typedef typename Derived::Scalar T;
  typedef Matrix<T, 3, 3> Mat33;

  Matrix<T, 4, 4> ret(Matrix<T, 4, 4>::Identity());

  const T theta = (p.template head<3>()).norm();
  if(theta > 1e-8) {
    T a = ::sin(theta);
    T b = 1.0 - ::cos(theta);
    T t_i = 1.0 / theta;
    const Mat33 S = t_i * MakeSkew( p.template head<3>() );
    const Mat33 S2 = S * S;
    const Mat33 I = Mat33::Identity();

    ret.template block<3,3>(0,0) = I + a*S + b*S2;
    ret.template block<3,1>(0,3) = (I + b*t_i*S + (theta - a)*t_i*S2)
        * p.template segment<3>(3);

  } else {
    ret.template block<3,1>(0,3) = p.template segment<3>(3);
  }

  return ret;
}

template <typename Derived> FORCE_INLINE
Eigen::Matrix<typename Derived::Scalar,6,1>
MatrixToTwist(const Eigen::MatrixBase<Derived>& M)
{
  using namespace Eigen;
  typedef typename Derived::Scalar T;
  typedef Matrix<T, 3, 3> Mat33;

  const auto R = M.template block<3,3>(0,0);
  const auto theta = ::acos( 0.5 * (R.trace()-1) );

  Matrix<T,6,1> ret;

  if(theta > 1e-10) {
    T s = ::sin(theta);
    T c = ::cos(theta);
    T a = s * (1.0 / theta);
    T b = (1.0 - c) * (1.0 / sq(theta));

    auto W = (theta/(2*s)) * (R - R.transpose());
    auto V = Mat33::Identity() - 0.5*W + (1/sq(theta))*(1-(a/(2*b)))*W*W;
    ret.template head<3>() = Matrix<T,3,1>(W(2,1), W(0,2), W(1,0));
    ret.template tail<3>() = V * M.template block<3,1>(0,3);
  } else {
    ret.template head<3>() = Matrix<T,3,1>::Zero();
    ret.template tail<3>() = M.template block<3,1>(0,3);
  }

  return ret;
}

/**
 * Converts a 3x3 rotation matrix (det(R) = +1) to Euler ZYX angles in radians
 */
template <class Derived> FORCE_INLINE
Eigen::Matrix<typename Derived::Scalar,3,1>
RotationMatrixToEulerAngles(const Eigen::MatrixBase<Derived>& R)
{
  typedef typename Derived::Scalar T;

  T eta = 1.0 / (std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0)));
  T rz = std::asin(eta * R(1,0));
  T ry = std::asin(-R(2,0));
  T rx = std::asin(eta * R(2,1));

  return Eigen::Matrix<T, 3, 1>(rx, ry, rz);
}

/**
 * Converts ZYX Euler angles to 3x3 in degrees to a 3x3 rotation
 */
template <typename T = double> FORCE_INLINE
Eigen::Matrix<T, 3, 3> EulerVectorToRotationMatrix(double a, double b, double c)
{
  a = deg2rad(a);
  b = deg2rad(b);
  c = deg2rad(c);

  auto ca = std::cos(a); auto sa = std::sin(a);
  auto cb = std::cos(b); auto sb = std::sin(b);
  auto cg = std::cos(c); auto sg = std::sin(c);

  return (Eigen::Matrix<T, 3, 3>() <<
      ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg,
      sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-ca*sg,
      -sb,   cb*sg,          cb*cg
      ).finished();
}


/**
 * converts Euler angles in degrees to rotation matrix
 *
 * TODO generic Eigen version
 */
template <typename T = double> FORCE_INLINE
Eigen::Matrix<T, 3, 3> EulerVectorToRotationMatrix(const Eigen::Matrix<T,3,1>& v)
{
  return EulerVectorToRotationMatrix(v[0], v[1], v[2]);
}

/**
 * converts Euler angles in degrees to rotation matrix
 *
 * TODO generic Eigen version
 */
template <typename T = double> FORCE_INLINE
Eigen::Matrix<T, 3, 3> EulerVectorToRotationMatrix(const Eigen::Matrix<T,1,3>& v)
{
  return EulerVectorToRotationMatrix(v[0], v[1], v[2]);
}


namespace se3 {

template <typename T> inline
Eigen::Matrix<T,4,4> exp(const Eigen::Matrix<T,6,1>& p)
{
  return TwistToMatrix(p);
}

template <typename T> inline
Eigen::Matrix<T,6,1> log(const Eigen::Matrix<T,4,4>& m)
{
  return MatrixToTwist(m);
}

template <class Derived> inline
Eigen::Matrix<typename Derived::Scalar, 4, 4>
Exp(const Eigen::MatrixBase<Derived>& v) { return TwistToMatrix(v); }

template <typename Derived> inline
Eigen::Matrix<typename Derived::Scalar,6,1>
Log(const Eigen::MatrixBase<Derived>& M) { return MatrixToTwist(M); }

}; // se3
}; // math

}; // bpvo

#endif // BPVO_MATH_UTILS_H
