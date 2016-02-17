#ifndef BPVO_UTILS_VIS_H
#define BPVO_UTILS_VIS_H

#include <limits>
#include <opencv2/core/core.hpp>

namespace bpvo {

class ImageFrame;
class StereoCalibration;

cv::Mat colorizeDisparity(const cv::Mat&, double min_val = 0.0, double max_val = 0.0);

cv::Mat overlayDisparity(const cv::Mat& I, const cv::Mat& D, double alpha,
                         double min_val = 0.0, double max_val = 0.0);

cv::Mat overlayDisparity(const ImageFrame* frame, double alpha = 0.5,
                         double min_val = 0.0, double max_val = 0.0);

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

