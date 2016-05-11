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

#ifndef BPVO_CENTRAL_DIFFERENCE_DESCRIPTOR_H
#define BPVO_CENTRAL_DIFFERENCE_DESCRIPTOR_H

#include <bpvo/dense_descriptor.h>
#include <opencv2/core/core.hpp>

namespace bpvo {

class CentralDifferenceDescriptor : public DenseDescriptor
{
 public:
  CentralDifferenceDescriptor(int radius = 3, float s1 = 0.75, float s2 = 1.75);
  CentralDifferenceDescriptor(const CentralDifferenceDescriptor&);
  virtual ~CentralDifferenceDescriptor();

  void compute(const cv::Mat&);

  inline int rows() const { return _rows; }
  inline int cols() const { return _cols; }
  inline int numChannels() const { return static_cast<int>(_channels.size()); }

  inline Pointer clone() const {
    return Pointer(new CentralDifferenceDescriptor(*this));
  }

  inline const cv::Mat& getChannel(int i) const { return _channels[i]; }

 private:
  int _radius;
  float _sigma_before, _sigma_after;
  int _rows, _cols;
  std::vector<cv::Mat> _channels;
}; // CentralDifferenceDescriptor

}; // bpvo

#endif // BPVO_CENTRAL_DIFFERENCE_DESCRIPTOR_H
