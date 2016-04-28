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

#ifndef BPVO_GRADIENT_DESCRIPTOR_H
#define BPVO_GRADIENT_DESCRIPTOR_H

#include <bpvo/dense_descriptor.h>
#include <opencv2/core/core.hpp>
#include <array>

namespace bpvo {

class GradientDescriptor : public DenseDescriptor
{
 public:
  /**
   * Intensity + gradient information
   *
   * \param sigma an optional smoothing prior to computing gradients
   */
  GradientDescriptor(float sigma = -1.0);

  GradientDescriptor(const GradientDescriptor& other);

  virtual ~GradientDescriptor();

  void compute(const cv::Mat&);

  inline const cv::Mat& getChannel(int i) const { return _channels[i]; }
  inline int numChannels() const { return 3; }
  inline int rows() const { return _rows; }
  inline int cols() const { return _cols; }

  inline void setSigma(float s) { _sigma = s; }

  inline Pointer clone() const
  {
    return Pointer(new GradientDescriptor(*this));
  }

  void copyTo(DenseDescriptor*) const;

 private:
  int _rows, _cols;
  float _sigma;
  std::array<cv::Mat,3> _channels;
}; // GradientDescriptor

}; // bpvo

#endif // BPVO_GRADIENT_DESCRIPTOR_H
