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

#ifndef BPVO_INTENSITY_DESCRIPTOR_H
#define BPVO_INTENSITY_DESCRIPTOR_H

#include <bpvo/dense_descriptor.h>
#include <bpvo/utils.h>
#include <opencv2/core/core.hpp>

namespace bpvo {

/**
 * The trivial form of the descriptor (grayscale image) converted to floating
 * point
 */
class IntensityDescriptor : public DenseDescriptor
{
 public:
  IntensityDescriptor() : DenseDescriptor() {}

  IntensityDescriptor(const IntensityDescriptor& o)
      : DenseDescriptor(o), _I(o._I.clone()) { }

  virtual ~IntensityDescriptor() {}

  void compute(const cv::Mat& src);

  void computeSaliencyMap(cv::Mat&) const;

  inline const cv::Mat& getChannel(int i) const
  {
    UNUSED(i);
    assert( i == 0 && "bad index" );

    return _I;
  }

  inline int numChannels() const { return 1; }

  inline int rows() const { return _I.rows; }
  inline int cols() const { return _I.cols; }

  inline Pointer clone() const
  {
    return Pointer(new IntensityDescriptor(*this));
  }

  void copyTo(DenseDescriptor*) const;

 protected:
  cv::Mat_<float> _I;
}; // IntensityDescriptor

}; // bpvo


#endif // BPVO_INTENSITY_DESCRIPTOR_H

