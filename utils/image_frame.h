#ifndef BPVO_UTILS_IMAGE_FRAME_H
#define BPVO_UTILS_IMAGE_FRAME_H

#include <opencv2/core/core.hpp>

namespace bpvo {

struct ImageFrame
{
  /** \return the image (grayscale) */
  virtual const cv::Mat& image() const = 0;

  /** \return the disparity (floating point) */
  virtual const cv::Mat& disparity() const = 0;

  virtual ~ImageFrame() {}
}; // ImageFrame

class StereoFrame : public ImageFrame
{
 public:
  StereoFrame();
  explicit StereoFrame(const cv::Mat& left, const cv::Mat& right);
  explicit StereoFrame(const cv::Mat& left, const cv::Mat& right, const cv::Mat& dmap);
  virtual ~StereoFrame();

  const cv::Mat& image() const;
  const cv::Mat& disparity() const;

  void setLeft(const cv::Mat&);
  void setRight(const cv::Mat&);
  void setDisparity(const cv::Mat&);

 private:
  cv::Mat _left;
  cv::Mat _right;
  cv::Mat _disparity;
}; // StereoFrame

class DisparityFrame : public ImageFrame
{
 public:
  DisparityFrame();
  explicit DisparityFrame(const cv::Mat& left, const cv::Mat& disparity);
  virtual ~DisparityFrame();

  const cv::Mat& image() const;
  const cv::Mat& disparity() const;

  void setImage(const cv::Mat&);
  void setDisparity(const cv::Mat&);

 private:
  cv::Mat _image;
  cv::Mat _disparity;

 protected:
  void convertDisparityToFloat();
}; // DisparityFrame


}; // bpvo

#endif // BPVO_UTILS_IMAGE_FRAME_H
