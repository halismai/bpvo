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

#include <bpvo/types.h>

namespace bpvo {

class PhotoError
{
 public:
  typedef EigenAlignedContainer<Point>::type PointVector;

 public:
  PhotoError(InterpolationType = InterpolationType::kLinear);
  ~PhotoError();

  /**
   * Initialize interpolation coefficients and tables
   *
   * \param pose    the camera pose K*[R t]
   * \param points  the vector of image points
   * \param valid   flags for valid points (the ones that project in the image)
   * \param rows    number of image rows
   * \param cols    number of image cols
   */
  void init(const Matrix34& pose, const PointVector& points, ValidVector& valid, int rows, int cols);

  /**
   * compute the vector of residuals by interpolating values from the current
   * image
   *
   * \param I0_ptr intensity values at the template image
   * \param I1     the input image
   * \param r_ptr  pointer to store residuals
   *
   * I0_ptr and r_ptr must have the same size
   */
  void run(const float* I0_ptr, const float* I1_ptr, float* r_ptr) const;

 protected:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // BilinearInterpolation

}; // bpvo
