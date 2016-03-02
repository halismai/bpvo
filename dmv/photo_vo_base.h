#ifndef BPVO_DMV_PHOTO_VO_BASE_H
#define BPVO_DMV_PHOTO_VO_BASE_H

#include <bpvo/types.h>
#include <dmv/world_point.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {
namespace dmv {

class PhotoVoBase
{
 public:
  struct Config
  {
    enum class PixelSelectorType
    {
      None, /** use all pixels */
      LocalAbsGradMax, /** local gradient maxima */
    }; // PixelSelectorType

    enum class DescriptorType
    {
      Pixel, /** a single pixel */
      Patch  /** an image patch */
    }; // DescriptorType

    //
    // Pixel selection options
    //
    PixelSelectorType pixelSelection = PixelSelectorType::LocalAbsGradMax;
    int nonMaxSuppRadius = 2; // 5x5 winddow
    int minSaliency = 1;

    //
    // optimization stuff
    //
    int maxIterations = 100;
    double parameterTolerance = 1e-6;
    double functionTolerance  = 1e-6;
    double gradientTolerance  = 1e-8;

    //
    // photo stuff
    //
    DescriptorType descriptorType = DescriptorType::Pixel;
  }; // Config

  struct Result
  {
    Mat_<double,4,4> T; // estimated pose
    std::vector<double> refinedDepth; //
  }; // Result

 public:
  /**
   */
  PhotoVoBase(const Mat_<double,3,3>& K, double b, Config& config);

  ~PhotoVoBase() {}

  void setTemplate(const cv::Mat& image, const cv::Mat& disparity);

  /**
   */
  virtual Result estimatePose(const cv::Mat& image) = 0;

 protected:
  Mat_<double,3,3> _K, _K_inv;
  double _baseline;
  Config _config;

 protected:
  typename EigenAlignedContainer<WorldPoint>::type _points;
}; // PhotoVoBase

}; // dmv
}; // bpvo

#endif // BPVO_DMV_PHOTO_VO_BASE_H
