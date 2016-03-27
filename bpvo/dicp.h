#ifndef BPVO_DICP_H
#define BPVO_DICP_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class DenseDescriptor;

class DirectDisparityIcp
{
 public:
  DirectDisparityIcp(int pyr_level, const Matrix33& K, float b);

  ~DirectDisparityIcp();

  /**
   * sets the data from the dense descriptor and the disparity
   */
  void setData(const DenseDescriptor*, const cv::Mat& D);


  void estimatePose(const DenseDescriptor*, const cv::Mat& D, const Matrix44& pose);

 private:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // DirectDisparityIcp

}; // bpvo

#endif // BPVO_DICP_H
