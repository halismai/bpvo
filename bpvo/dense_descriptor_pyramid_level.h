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

#ifndef BPVO_DENSE_DESCRIPTOR_PYRAMID_LEVEL_H
#define BPVO_DENSE_DESCRIPTOR_PYRAMID_LEVEL_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class DenseDescriptor;

/**
 */
class DenseDescriptorPyramidLevel
{
 public:
  /**
   */
  explicit DenseDescriptorPyramidLevel(const AlgorithmParameters&, int pyr_level);

  DenseDescriptorPyramidLevel(const DenseDescriptorPyramidLevel&) = delete;
  DenseDescriptorPyramidLevel& operator=(const DenseDescriptorPyramidLevel&) = delete;

  DenseDescriptorPyramidLevel(DenseDescriptorPyramidLevel&&);
  DenseDescriptorPyramidLevel& operator=(DenseDescriptorPyramidLevel&&);


  /**
   * set the image for the descriptor. The descriptor will be computed on-demand
   * later, unless compute is set to true
   */
  void setImage(const cv::Mat&, bool compute_now = false);

  /**
   * get the descriptor data
   *
   * setImage must be called before hand
   */
  const cv::Mat& getDescriptorData(bool force_recompute = false);

 private:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // DenseDescriptorPyramidLevel

}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_PYRAMID_LEVEL_H

