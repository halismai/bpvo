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

#ifndef BPVO_CHANNELS_H
#define BPVO_CHANNELS_H

#include <bpvo/types.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <array>

namespace bpvo {

class RawIntensity
{
 public:
  static constexpr int NumChannels = 1;
  typedef typename AlignedVector<float>::type PixelVector;
  typedef float ChannelDataType;

 public:
  RawIntensity(float = 0.0f, float = 0.0f);
  explicit RawIntensity(const cv::Mat& I);

  inline int size() const { return NumChannels; }

  inline const cv::Mat_<ChannelDataType>& operator[](int) const {
    return _I;
  }

  inline const ChannelDataType* channelData(int) const {
    return _I.ptr<const ChannelDataType>();
  }

  void compute(const cv::Mat&);

  void computeSaliencyMap(cv::Mat_<float>&) const;

  inline int rows() const { return _I.rows; }
  inline int cols() const { return _I.cols; }

 protected:
  cv::Mat_<ChannelDataType> _I;
}; // RawIntensity


class BitPlanes
{
 public:
  static constexpr int NumChannels = 8;
  typedef typename AlignedVector<float>::type PixelVector;
  typedef float ChannelDataType;
  typedef std::array<cv::Mat_<ChannelDataType>,NumChannels> ChannelsArray;

 public:
  /**
   */
  inline BitPlanes(float s1 = 0.5f, float s2 = 0.5f)
      : _sigma_ct(s1), _sigma_bp(s2) {}

  inline BitPlanes(const cv::Mat& I, float s1 = 0.5f, float s2 = 0.5f)
      : BitPlanes(s1, s2) { compute(I); }

  inline int size() const { return NumChannels; }

  inline const cv::Mat_<ChannelDataType>& operator[](int i) const {
    return _channels[i];
  }

  inline const ChannelDataType* channelData(int c) const {
    return _channels[c].ptr<const ChannelDataType>();
  }

  inline void setSigmaCensus(float s) { _sigma_ct = s; }
  inline void setSigmaBitPlanes(float s) { _sigma_bp = s; }

  inline const float& sigmaCensus() const { return _sigma_ct; }
  inline const float& sigmaBitPlanes() const {  return _sigma_bp; }

  void computeSaliencyMap(cv::Mat_<float>&) const;

  void compute(const cv::Mat&);

  inline int rows() const { return _channels.front().rows; }
  inline int cols() const { return _channels.front().cols; }

 protected:
  float _sigma_ct = 0.0f;
  float _sigma_bp = 0.0f;
  ChannelsArray _channels;
}; // BitPlanes


}; // bpvo


#endif // BPVO_CHANNELS_H

