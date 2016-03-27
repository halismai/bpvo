#ifndef SPS_H
#define SPS_H

namespace cv {
class Mat;
}; // cv


class SpsStereo
{
 public:
  struct Config
  {
    int numInnerIterations;
    int numOuterIterations;

    double outputDisparityFactor;
    double positionWeight;
    double disparityWeight;
    double boundaryLengthWeight;
    double smoothnessWeight;
    double inlierThreshold;

    double hingePenalty;
    double occlusionPenalty;
    double impossiblePenalty;

    Config();
  }; // Config

 public:
  SpsStereo(Config = Config());
  ~SpsStereo();

  inline const Config& config() const { return _config; }
  inline       Config& config()       { return _config; }

  void compute(const cv::Mat&, const cv::Mat&, cv::Mat&);


 private:
  Config _config;
  struct Impl;
  Impl* _impl;
}; // SpsStereo


#endif // SPS_H
