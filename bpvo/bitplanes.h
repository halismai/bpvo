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

#ifndef BPVO_BITPLANES_H
#define BPVO_BITPLANES_H

#include <array>
#include <opencv2/core/core.hpp>

namespace bpvo {

class BitPlanesData
{
  typedef std::array<cv::Mat,8> ChannelsContainer;

 public:
  /**
   * \param s1 sigmaPriorToCensusTransform
   * \param s2 sigmaBitPlanes
   */
  BitPlanesData(float s1, float s2);


  /**
   * \param s1 sigmaPriorToCensusTransform
   * \param s2 sigmaBitPlanes
   * \params I the image (grayscale 8-bit)
   */
  explicit BitPlanesData(float s1, float s2, const cv::Mat&);

  /**
   */
  BitPlanesData& computeChannels(const cv::Mat&);

  /**
   * compute the gradientAbsMag
   */
  void computeGradientAbsMag();

  /**
   * \return the gradientAbsMag
   */
  inline const cv::Mat& gradientAbsMag() const noexcept { return _gmag; }

  /**
   */
  static inline BitPlanesData Compute(const cv::Mat& I, float s1, float s2) {
    return BitPlanesData(s1, s2, I);
  }

  inline const ChannelsContainer& channels() const noexcept { return _channels; }

  inline const cv::Mat& operator[](int i) const {
    return _channels[i];
  }

  template <int i> inline
  const cv::Mat& getChannel() const noexcept {
    static_assert(i >= 0 && i < 8, "bad index");
    return _channels[i];
  }

  inline size_t size() const noexcept { return _channels.size(); }

 private:
  float _sigma_ct;
  float _sigma_bp;

  /**
   * the channels
   */
  ChannelsContainer _channels;

  /**
   * an image of the absolute gradient magnitude for all channels.
   * We use this to determine if a pixel is suitable for alignment. Pixels with
   * no gradient no effect on the optimization
   */
  cv::Mat _gmag;


  friend class BitPlanesChannelMaker;

}; // BitPlanesData

BitPlanesData computeBitPlanes(const cv::Mat&, float s1, float s2);

}; // bpvo

#endif // BPVO_BITPLANES_H
