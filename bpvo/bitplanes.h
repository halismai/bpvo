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
  std::array<cv::Mat,8> I;

  /**
   * gradients. Each pixel has [Ix Iy] 2 elements per pixel CV_32FC2
   */
  std::array<cv::Mat,8> G;

  /**
   * compute the gradients at channel i
   */
  void computeGradients(int i);
}; // BitPlanesData

BitPlanesData computeBitPlanes(const cv::Mat&, float s1, float s2);

}; // bpvo

#endif // BPVO_BITPLANES_H
