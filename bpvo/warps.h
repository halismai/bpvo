#ifndef BPVO_WARPS_H
#define BPVO_WARPS_H

#include <bpvo/types.h>
#include <bpvo/math_utils.h>

namespace bpvo {

namespace detail {
template <class> struct warp_traits;
}; // detail

class RigidBodyWarp;
namespace detail {
template <> struct warp_traits<RigidBodyWarp>
{
  typedef bpvo::Point Point;
  typedef Eigen::Matrix<float,2,1> ImagePoint;
  typedef Eigen::Matrix<float,1,6> Jacobian;
  typedef Eigen::Matrix<float,2,6> WarpJacobian;

  typedef typename EigenAlignedContainer<Jacobian>::type      JacobianVector;
  typedef typename EigenAlignedContainer<Point>::type         PointVector;
  typedef typename EigenAlignedContainer<WarpJacobian>::type  WarpJacobianVector;
}; // warp_traits

}; // details


class RigidBodyWarp
{
 public:
  typedef detail::warp_traits<RigidBodyWarp> Traits;

  typedef typename Traits::Point              Point;
  typedef typename Traits::ImagePoint         ImagePoint;
  typedef typename Traits::Jacobian           Jacobian;
  typedef typename Traits::WarpJacobian       WarpJacobian;
  typedef typename Traits::PointVector        PointVector;
  typedef typename Traits::JacobianVector     JacobianVector;
  typedef typename Traits::WarpJacobianVector WarpJacobianVector;

 public:
  RigidBodyWarp(const Matrix33& K, float b);

  Point makePoint(float x, float y, float d) const;

  void setNormalization(const Matrix44&);
  void setNormalization(const PointVector& points);

  WarpJacobian warpJacobianAtZero(const Point&) const;

  inline const Matrix33& K() const { return _K; }

  inline void setPose(const Matrix44& T) { _P = _K * T.block<3,4>(0,0); }

  inline ImagePoint operator()(const Point& X) const
  {
    Eigen::Vector3f x = _P * X;
    float z_i = 1.0f / x[2];
    return ImagePoint(x[0]*z_i, x[1]*z_i);
  }

  inline Matrix44 scalePose(const Matrix44& T) const
  {
    return _T_inv * T * _T;
  }

  template <class Derived> inline
  Matrix44 paramsToPose(const Eigen::MatrixBase<Derived>& p) const
  {
    return _T_inv * math::TwistToMatrix(p) * _T;
  }


 protected:
  Matrix33 _K;
  float _b;

  Matrix34 _P;

  // normalization
  Matrix44 _T;
  Matrix44 _T_inv;
}; // RigidBodyWarp

Matrix44 HartlyNormalization(const typename RigidBodyWarp::PointVector& pts);

}; // bpvo

#endif // BPVO_WARPS_H
