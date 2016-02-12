#ifndef BPVO_UTILS_VIS_H
#define BPVO_UTILS_VIS_H

#include <limits>
#include <opencv2/core/core.hpp>

namespace bpvo {

class ImageFrame;

cv::Mat colorizeDisparity(const cv::Mat&, double min_val=std::numeric_limits<double>::quiet_NaN(),
                          double max_val = std::numeric_limits<double>::quiet_NaN());

cv::Mat overlayDisparity(const cv::Mat& I, const cv::Mat& D, double alpha,
                         double min_val, double max_val);

cv::Mat overlayDisparity(const ImageFrame* frame, double alpha = 0.5f,
                         double min_val = std::numeric_limits<double>::quiet_NaN(),
                         double max_val = std::numeric_limits<double>::quiet_NaN());


class DisparityPointCloudViewer
{
}; //

}; // bpvo

#endif // BPVO_UTILS_VIS_H

