#ifndef BPVO_UTILS_STEREO_ALGORITHM_H
#define BPVO_UTILS_STEREO_ALGORITHM_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class ConfigFile;

class StereoAlgorithm
{
 public:
  StereoAlgorithm(const ConfigFile&);
  StereoAlgorithm(std::string conf_fn);

  virtual ~StereoAlgorithm();

  /**
   * \param left the left image
   * \param right the right image
   * \param dmap disparity map
   */
  void run(const cv::Mat& left, const cv::Mat& right, cv::Mat& dmap);

  /**
   * \return the invalid value as a floating point number
   */
  float getInvalidValue() const;

 protected:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // StereoAlgorithm

}; // bpvo

#endif // BPVO_UTILS_STEREO_ALGORITHM_H
