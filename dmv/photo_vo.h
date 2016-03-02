#ifndef BPVO_DMV_PHOTO_VO
#define BPVO_DMV_PHOTO_VO

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {
namespace dmv {

class PhotoVo
{
 public:

  struct Config
  {
    int nonMaxSuppRadius = 1;
    int maxIterations = 200;
    short minSaliency = 1;

    bool withRobust = true;

    Config() {}
  }; // Config

  struct Result
  {
    Matrix44 pose;
    int numIterations;
  }; // Result

  static constexpr double PixelScale = 1.0 / 255.0;

 public:
  PhotoVo(const Eigen::Matrix<double,3,3>& K, double b, Config = Config());

  void setTemplate(const cv::Mat& I, const cv::Mat& D);

  Result estimatePose(const cv::Mat& I, const Matrix44& T_init = Matrix44::Identity()) const;

  inline size_t numPoints() const
  {
    return _X0.size();
  }

 private:
  Eigen::Matrix<double,3,3> _K;
  double _baseline;
  Config _config;

  typename AlignedVector<double>::type _I0;
  typename EigenAlignedContainer<Eigen::Matrix<double,3,1>>::type _X0;
}; // PhotoVo

}; // dmv
}; // bpvo

#endif // BPVO_DMV_PHOTO_VO
