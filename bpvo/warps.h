#ifndef BPVO_WARPS_H
#define BPVO_WARPS_H

#include <bpvo/types.h>

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

  typedef typename Traits::Point Point;
  typedef typename Traits::ImagePoint ImagePoint;
  typedef typename Traits::Jacobian Jacobian;
  typedef typename Traits::WarpJacobian WarpJacobian;
  typedef typename Traits::PointVector PointVector;
  typedef typename Traits::JacobianVector JacobianVector;
  typedef typename Traits::WarpJacobianVector WarpJacobianVector;

 public:
  RigidBodyWarp(const Matrix33& K, float b);

  Point makePoint(float x, float y, float d) const;

  WarpJacobian warpJacobianAtZero(const Point&) const;

  inline const Matrix33& K() const { return _K; }

  inline void setPose(const Matrix44& T) { _P = _K * T.block<3,4>(0,0); }

  inline ImagePoint operator()(const Point& X) const
  {
    Eigen::Vector3f xw = _P * X;
    return ImagePoint(xw[0]/xw[2], xw[1]/xw[2]);
  }

 protected:
  Matrix33 _K;
  float _b;

  Matrix34 _P;
}; // RigidBodyWarp

}; // bpvo

#endif // BPVO_WARPS_H
