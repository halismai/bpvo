#include  "utils/image_frame.h"

namespace bpvo {

StereoFrame::StereoFrame() {}
StereoFrame::~StereoFrame() {}
StereoFrame::StereoFrame(const cv::Mat& left, const cv::Mat& right)
    : _left(left), _right(right) {}
StereoFrame::StereoFrame(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity)
    : _left(left), _right(right), _disparity(disparity) {}

const cv::Mat& StereoFrame::image() const { return _left; }
const cv::Mat& StereoFrame::disparity() const { return _disparity;  }

void StereoFrame::setLeft(const cv::Mat& I) { _left = I; }
void StereoFrame::setRight(const cv::Mat& I) { _right = I; }
void StereoFrame::setDisparity(const cv::Mat& D) { _disparity = D; }


DisparityFrame::DisparityFrame() {}

DisparityFrame::DisparityFrame(const cv::Mat& image, const cv::Mat& disparity)
  : _image(image), _disparity(disparity)
{
  convertDisparityToFloat();
}

DisparityFrame::~DisparityFrame() {}

const cv::Mat& DisparityFrame::image() const { return _image; }
const cv::Mat& DisparityFrame::disparity() const { return _disparity; }

void DisparityFrame::setImage(const cv::Mat& image)
{
  _image = image;
}

void DisparityFrame::setDisparity(const cv::Mat& disparity)
{
  _disparity = disparity;
  convertDisparityToFloat();
}

void DisparityFrame::convertDisparityToFloat()
{
  assert( _disparity.channels() == 1 );
  if(_disparity.type() != cv::DataType<float>::type)
    _disparity.convertTo(_disparity, CV_32FC1, 1.0/16.0, 0.0);
}

}; // bpvo

