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

#ifndef BPVO_TEMPLATE_DATA_H
#define BPVO_TEMPLATE_DATA_H

#include <bpvo/rigid_body_warp.h>
#include <bpvo/photo_error.h>
#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class DenseDescriptor;

// TODO this class should be storage only, the warp and PhotoError should be
// taken outside
class TemplateData
{
 public:
  typedef RigidBodyWarp WarpType;
  typedef WarpType Warp;
  typedef typename WarpType::Point Point;
  typedef typename WarpType::Jacobian Jacobian;
  typedef typename WarpType::PointVector PointVector;
  typedef typename WarpType::JacobianVector JacobianVector;

  typedef ResidualsVector PixelVector;

 public:
  /**
   * \param K the intrinsics matrix
   * \param b the stereo baseline
   * \param AlgorithmParameters
   */
  TemplateData(int pyr_level, const Matrix33& K, float baseline, const AlgorithmParameters&);

  /**
   * Sets the template data using the computed DenseDescriptor along with the
   * disparity map
   */
  void setData(const DenseDescriptor*, const cv::Mat& disparity);

  void computeResiduals(const DenseDescriptor*, const Matrix44& pose,
                        ResidualsVector&, ValidVector&) const;

  inline int numPixels() const { return (int) _pixels.size(); }
  inline int numPoints() const { return (int) _points.size(); }

  inline const PointVector& points() const { return _points; }
  inline const PixelVector& pixels() const { return _pixels; }
  inline const JacobianVector& jacobians() const { return _jacobians; }

  inline const Warp& warp() const { return _warp; }

 private:
  int _pyr_level;
  AlgorithmParameters _params;
  mutable RigidBodyWarp _warp; // should take the warp outside of this class

  JacobianVector _jacobians;
  PointVector _points;
  PixelVector _pixels;

  mutable PhotoError _photo_error;
}; // TemplateData

}; // bpvo

#endif // BPVO_TEMPLATE_DATA_H

