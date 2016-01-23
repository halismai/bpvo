#ifndef BPVO_VO_H
#define BPVO_VO_H

#include <bpvo/types.h>

namespace bpvo {

class VisualOdometry
{
 public:
  /**
   * \param ImageSize
   * \param AlgorithmParameters
   */
  VisualOdometry(ImageSize, AlgorithmParameters = AlgorithmParameters());

  /**
   */
  ~VisualOdometry();

  /**
   * Estimate the motion of the image wrt. to the previously added frame
   *
   * \param image pointer to the image data
   * \param disparity pionter to disparity map
   *
   * Both the image & disparity must have the same size
   *
   * \retun Result
   */
  Result addFrame(const uint8_t* image, const float* disparity);

 private:
  struct Impl;
  Impl* _impl;
}; // VisualOdometry

}; // bpvo

#endif
