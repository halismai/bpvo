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

#ifndef BPVO_VO_H
#define BPVO_VO_H

#include <bpvo/types.h>

namespace bpvo {

class Trajectory;

class VisualOdometry
{
  typedef typename EigenAlignedContainer<Point>::type PointVector;

 public:
  /**
   * \param K the intrinsic calibration of the rectified image
   * \param b the stereo baseline
   * \param ImageSize
   * \param AlgorithmParameters
   */
  VisualOdometry(const Matrix33& K, float baseline, ImageSize,
                 const AlgorithmParameters& = AlgorithmParameters());

  /**
   * create VisualOdometry from CalibrationT object
   */
  template <class CalibrationT> inline
  VisualOdometry(const CalibrationT& calib, ImageSize image_size,
                 const AlgorithmParameters& params = AlgorithmParameters()) :
      VisualOdometry(calib.K, calib.baseline, image_size, params) {}

  /**
   * create VisualOdometry from DataLoaderT object
   */
  template <class DataLoaderT> inline
  VisualOdometry(const DataLoaderT* data_loader,
                 const AlgorithmParameters& params = AlgorithmParameters()) :
      VisualOdometry(data_loader->calibration(), data_loader->imageSize(), params) {}

  /**
   */
  ~VisualOdometry();

  /**
   * Estimate the motion of the image wrt. to the previously added frame
   *
   * \param image pointer to the image data
   * \param disparity pionter to disparity map
   *
   * Both the image & disparity must have the same size
   *
   * \retun Result
   */
  Result addFrame(const uint8_t* image, const float* disparity);

  template <class FramePointer> inline
  Result addFrame(const FramePointer& frame) {
    return this->addFrame(frame->image().template ptr<const uint8_t>(),
                          frame->disparity().template ptr<const float>());
  }


  /**
   * \return the number of points at the specified pyramid level
   * The is the same as pointsAtLevel(level).size()
   */
  int numPointsAtLevel(int level = -1) const;

  /**
   * \return all the points at the specified level. This will always return
   * points if you ask it to (given that there is template data)
   */
  const PointVector& pointsAtLevel(int level = -1) const;


  /**
   * \return the trajectory thus far
   */
  const Trajectory& trajectory() const;

 private:
  class Impl;
  Impl* _impl;
}; // VisualOdometry

}; // bpvo

#endif
