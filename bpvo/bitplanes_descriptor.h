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

#ifndef BPVO_BITPLANES_DESCRIPTOR_H
#define BPVO_BITPLANES_DESCRIPTOR_H

#include <bpvo/dense_descriptor.h>
#include <opencv2/core/core.hpp>
#include <array>

namespace bpvo {

class BitPlanesDescriptor : public DenseDescriptor
{
 public:
  /**
   * Creates the dense bit-planes descriptor
   *
   * \param s0 std. dev of a Gaussian to apply to the input image before
   * computing the census signatures. If the value is <= 0, no smoothing will be
   * applied.
   *
   * Typical values for s0 are within 0.25 - 1.0, depending on the noise in the
   * data
   *
   *
   * \param s1 std. dev of a Gaussian to apply to each of the bit-planes. If the
   * value is <= 0, no smoothing will be applied (the bit-planes will be
   * binary). NOTE: we will scale the bits by 255 so that thresholds for
   * IntensityDescriptor work the same with BitPlanes.
   *
   * Typical values for s1 are within 0.5 - 1.5 depending no how much smoothing
   * you want
   *
   * NOTE: preforming this smoothing over the 8 bit-planes will be expensive,
   * but not so bad (depending on how many cores you have)
   */
  BitPlanesDescriptor(float s0 = 0.5f, float s1 = -1.0);

  virtual ~BitPlanesDescriptor();

  BitPlanesDescriptor(const BitPlanesDescriptor& other)
      : DenseDescriptor(other)
      , _rows(other._rows), _cols(other._cols), _sigma_ct(other._sigma_ct)
      , _sigma_bp(other._sigma_bp), _channels(other._channels) {}

  void compute(const cv::Mat&);

  inline const cv::Mat& getChannel(int i) const  { return _channels[i]; }

  inline int numChannels() const { return 8; }
  inline int rows() const { return _rows; }
  inline int cols() const { return _cols; }

  inline void setSigmaPriorToCensus(float s) { _sigma_ct = s; }
  inline void setSigmaBitPlanes(float s) { _sigma_bp = s; }

  inline Pointer clone() const
  {
    return Pointer(new BitPlanesDescriptor(*this));
  }

 private:
  int _rows, _cols;
  float _sigma_ct, _sigma_bp;
  std::array<cv::Mat,8> _channels;
}; // BitPlanesDescriptor

}; // bpvo


#endif // BPVO_BITPLANES_DESCRIPTOR_H
