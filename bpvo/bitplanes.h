#ifndef BPVO_BITPLANES_H
#define BPVO_BITPLANES_H

#include <array>
#include <opencv2/core.hpp>

namespace bpvo {

struct BitPlanesData
{
  /**
   * the channels
   */
  std::array<cv::Mat,8> cn;

  /**
   * an image of the absolute gradient magnitude for all channels.
   * We use this to determine if a pixel is suitable for alignment. Pixels with
   * no gradient no effect on the optimization
   */
  cv::Mat gradientAbsMag;

  /**
   */
  void computeGradientAbsMag();
}; // BitPlanesData

BitPlanesData computeBitPlanes(const cv::Mat&, float s1, float s2);

}; // bpvo

#endif // BPVO_BITPLANES_H
