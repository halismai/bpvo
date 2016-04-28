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

#ifndef BPVO_DENSE_DESCRIPTOR_H
#define BPVO_DENSE_DESCRIPTOR_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

/**
 * Base class for all dense descriptors
 */
class DenseDescriptor
{
 public:
  typedef UniquePointer<DenseDescriptor> Pointer;

 public:
  DenseDescriptor(const DenseDescriptor&) = default;
  virtual ~DenseDescriptor();

  DenseDescriptor() = default;
  DenseDescriptor(DenseDescriptor&&) noexcept = default;

  DenseDescriptor& operator=(const DenseDescriptor&) = default;
  DenseDescriptor& operator=(DenseDescriptor&&) noexcept = default;

  /**
   * Computes the channels/descriptors given the input image
   */
  virtual void compute(const cv::Mat& image) = 0;

  /**
   * Computes the saliency map and stores it in smap.
   *
   * compute() should be called prior to calling this function
   */
  virtual void computeSaliencyMap(cv::Mat& smap) const;

  /**
   * \return the i-th channel, the index must be less than the number of
   * channels for the descriptor
   */
  virtual const cv::Mat& getChannel(int i) const = 0;

  /**
   * \return a deep copy of the object
   */
  virtual Pointer clone() const = 0;

  /**
   * Copies that data into another DenseDescriptor
   */
  virtual void copyTo(DenseDescriptor* other) const;

  /**
   * \return the number of channels
   */
  virtual int numChannels() const = 0;

  /**
   * \return number of rows of the descriptor. The same across all channels
   */
  virtual int rows() const = 0;

  /**
   * \return number of columns of the descriptor. The same across all channels
   */
  virtual int cols() const = 0;

  static DenseDescriptor* Create(const AlgorithmParameters&, int pyr_level = 0);
}; // DenseDescriptor


}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_H
