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

#ifndef BPVO_WARPS_H
#define BPVO_WARPS_H

#include <bpvo/types.h>
#include <bpvo/math_utils.h>
#include <iostream>

namespace bpvo {

namespace detail {
template <class> struct warp_traits;
}; // detail

class RigidBodyWarp;
class DisparitySpaceWarp;

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

template <> struct warp_traits<DisparitySpaceWarp> :
  public warp_traits<RigidBodyWarp> {};

}; // detail

/**
 * Apply Hartely's normalization on the points
 * \return a scaling matrix such that T * X results in points with zero mean and
 * unit std. dev on a 3-sphere
 */
Matrix44 HartlyNormalization(const typename detail::warp_traits<RigidBodyWarp>::PointVector& pts);


template <class Derived>
class WarpBase
{
 public:
  typedef detail::warp_traits<Derived> Traits;

  typedef typename Traits::Point          Point;
  typedef typename Traits::ImagePoint     ImagePoint;
  typedef typename Traits::Jacobian       Jacobian;
  typedef typename Traits::WarpJacobian   WarpJacobian;
  typedef typename Traits::JacobianVector JacobianVector;
  typedef typename Traits::PointVector    PointVector;
  typedef typename Traits::WarpJacobianVector WarpJacobianVector;

  static constexpr int DOF = Jacobian::ColsAtCompileTime;
  static_assert( DOF != Eigen::Dynamic, "DOF's must be known at compile time");

 public:
  /**
   * create a point from x,y,d
   *
   * \param x, y coordinates in the image plane
   * \param d    the disparity
   */
  inline Point makePoint(float x, float y, float d) const
  {
    return derived()->makePoint(x, y, d);
  }

  /**
   * \return the intrinsic calibration matrix
   */
  inline const Matrix33& K() const { return derived()->K(); }

  /**
   * set the camera pose for the warp. This must be called before warping points
   */
  inline void setPose(const Matrix44& T)
  {
    derived()->setPose(T);
  }

  /**
   * set a normalizatioon matrix for the points. Effects of normalization will
   * propagate throw to the Jacobian computation
   */
  inline void setNormalization(const Matrix44& T)
  {
    derived()->setNormalization(T);
  }

  /**
   * set the normalization using the vector of points
   */
  inline void setNormalization(const PointVector& points)
  {
    setNormalization(HartlyNormalization(points));
  }

  /**
   * compute the Jacobian at zero given a point and image gradient [Ix, Iy]
   */
  template <class ... Args> inline
  void jacobian(const Point& p, float Ix, float Iy, float* J, Args&& ... args) const
  {
    derived()->jacobian(p, Ix, Iy, J, args...);
  }

  /**
   * converts a vector of parameters to a 4x4 pose matrix
   */
  template <class EigenType> inline
  Matrix44 paramsToPose(const Eigen::MatrixBase<EigenType>& p) const
  {
    return derived()->paramsToPose(p);
  }

  /**
   * transforms the the point given the current camera pose
   */
  inline ImagePoint operator()(const Point& p) const
  {
    return derived()->operator()(p);
  }

  /**
   * \return the 2D image point from thhe input
   * used when colorizing the point clouds
   */
  inline ImagePoint getImagePoint(const Point& p) const
  {
    return derived()->getImagePoint(p);
  }

  inline const Derived* derived() const { return static_cast<const Derived*>(this); }
  inline Derived* derived() { return static_cast<Derived*>(this); }
}; // WarpBase



}; // bpvo

#endif // BPVO_WARPS_H

