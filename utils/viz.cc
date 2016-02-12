#include "utils/viz.h"
#include "utils/image_frame.h"

#include <opencv2/contrib/contrib.hpp>

namespace bpvo {

cv::Mat colorizeDisparity(const cv::Mat& D, double min_val, double max_val)
{
  cv::Mat ret(D);
  if(std::isnan(min_val) || std::isnan(max_val))
    cv::minMaxLoc(ret, &min_val, &max_val);

  ret = 255.0 * ((ret - min_val) / (max_val - min_val));
  ret.convertTo(ret, CV_8U);
  cv::applyColorMap(ret, ret, cv::COLORMAP_JET);

  for(int r = 0; r < ret.rows; ++r)
    for(int c = 0; c < ret.cols; ++c) {
      if(D.at<float>(r,c) < 1e-3f) {
        ret.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,0);
      }
    }

  return ret;
}

cv::Mat overlayDisparity(const cv::Mat& I, const cv::Mat& D, double alpha, double min_val, double max_val)
{
  cv::Mat image = I;
  if(image.channels() != 3)
    cv::cvtColor(I, image, cv::COLOR_GRAY2BGR);

  cv::Mat dst;
  cv::addWeighted(image, alpha, colorizeDisparity(D, min_val, max_val), 1.0-alpha, 0.0, dst);

  return dst;
}

cv::Mat overlayDisparity(const ImageFrame* frame, double alpha, double min_val, double max_val)
{
  return overlayDisparity(frame->image(), frame->disparity(), alpha, min_val, max_val);
}

} // bpvo

