#ifndef BPVO_UTILS_VIS_H
#define BPVO_UTILS_VIS_H

#include <limits>
#include <opencv2/core/core.hpp>

namespace bpvo {

struct StereoCalibration;

void colorizeDisparity(const cv::Mat& src, cv::Mat& dst, double min_d = 0, double num_d = -1);

inline cv::Mat colorizeDisparity(const cv::Mat& src, double min_d = 0, double num_d = -1)
{
  cv::Mat ret;
  colorizeDisparity(src, ret, min_d, num_d);
  return ret;
}

void overlayDisparity(const cv::Mat& I, const cv::Mat& D, cv::Mat& dst,
                      double alpha = 0.5, double min_d = 0.0, double num_d = -1);

// TODO
class DisparityPointCloudViewer
{
 public:
  /**
   * initialize the viewer with a StereoCalibration
   */
  DisparityPointCloudViewer(const StereoCalibration&);

  virtual ~DisparityPointCloudViewer();

  /**
   * add a disparity to the viewer. It will be triangulated and added to the display queue
   */
  void addDisparity(const cv::Mat&);

  /**
   * \return true if the viewer is running
   */
  bool isRunning() const;

  /**
   * ask the viewer to stop
   */
  void stop();

 protected:
  struct Impl;
  Impl* _impl;
}; //

}; // bpvo

#endif // BPVO_UTILS_VIS_H

