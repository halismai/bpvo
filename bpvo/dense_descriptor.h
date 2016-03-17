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

class DenseDescriptor
{
 public:
  typedef UniquePointer<DenseDescriptor> Pointer;

 public:
  DenseDescriptor();
  DenseDescriptor(const DenseDescriptor&);
  virtual ~DenseDescriptor();

  /**
   * computes the channels/descriptors
   */
  virtual void compute(const cv::Mat&) = 0;

  /**
   * computes the saliency map
   */
  virtual void computeSaliencyMap(cv::Mat&) const = 0;

  /**
   * \return the i-th channel
   */
  virtual const cv::Mat& getChannel(int i) const = 0;

  /**
   */
  virtual Pointer clone() const = 0;

  /**
   * \return the number of channels
   */
  virtual int numChannels() const = 0;

  virtual int rows() const = 0;
  virtual int cols() const = 0;

  inline void setHasData(bool v) { _has_data = v;}

  inline bool hasData() const { return _has_data; }

 protected:
  bool _has_data = false;
}; // DenseDescriptor


}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_H
