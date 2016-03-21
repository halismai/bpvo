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

#ifndef BPVO_DENSE_DESCRIPTOR_PYRAMID_H
#define BPVO_DENSE_DESCRIPTOR_PYRAMID_H

#include <bpvo/types.h>
namespace cv {
class Mat;
}; // cv

namespace bpvo {

class ImagePyramid;
class DenseDescriptor;

/**
 */
class DenseDescriptorPyramid
{
 public:
  /**
   * Allocates memory for p.numPyramidLevels
   */
  DenseDescriptorPyramid(const AlgorithmParameters& p);
  ~DenseDescriptorPyramid(); // for the satisfaction of UniquePointer

  DenseDescriptorPyramid(DenseDescriptorPyramid&&) noexcept;
  DenseDescriptorPyramid& operator=(DenseDescriptorPyramid&&) = default;

  DenseDescriptorPyramid(const DenseDescriptorPyramid&) = delete;
  DenseDescriptorPyramid& operator=(const DenseDescriptorPyramid&) = default;

  void copyTo(DenseDescriptorPyramid&) const;

  /**
   * must be called before accessing the descriptors
   */
  void init(const cv::Mat&);

  /**
   * init using a pre-computed image pyramid
   */
  void init(const ImagePyramid&);

  /**
   * \return the descriptor at level 'i'
   */
  const DenseDescriptor* operator[](size_t i) const;
  DenseDescriptor* operator[](size_t i);

  /**
   * \return the number of pyramid levels
   */
  int size() const;

 private:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // DenseDescriptorPyramid
}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_PYRAMID_H

