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

#ifndef BPVO_LATCH_DESCRIPTOR_H
#define BPVO_LATCH_DESCRIPTOR_H

#include <bpvo/dense_descriptor.h>
#include <opencv2/core/core.hpp>
#include <vector>

namespace bpvo {

class LATCHDescriptorExtractorImpl;

class LatchDescriptor : public DenseDescriptor
{
 public:
  LatchDescriptor(int bytes = 32, bool rotationInvariance = false, int half_ssd_size = 3);
  LatchDescriptor(const LatchDescriptor&);
  virtual ~LatchDescriptor();

  void compute(const cv::Mat&);

  inline int rows() const { return _rows; }
  inline int cols() const { return _cols; }
  inline int numChannels() const { return (int) _channels.size(); }

  inline Pointer clone() const {
    return Pointer(new LatchDescriptor(*this));
  }

  inline const cv::Mat& getChannel(int i) const { return _channels[i]; }

 private:
  UniquePointer<LATCHDescriptorExtractorImpl> _impl;
  int _rows, _cols;
  std::vector<cv::Mat> _channels;

  cv::Mat _buffer;
}; // LatchDescriptor

}; // bpvo

#endif // BPVO_LATCH_DESCRIPTOR_H

