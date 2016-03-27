#ifndef SGM_H
#define SGM_H

namespace cv {
class Mat;
}; // cv

class SgmStereo
{
 public:
  struct Config
  {
    int numberOfDisparities;
    int sobelCapValue;
    int censusRadius;
    int windowRadius;
    int smoothnessPenaltySmall;
    int smoothnessPenaltyLarge;
    int consistencyThreshold;

    double disparityFactor;
    double censusWeightFactor;

    Config();
  }; // Config

 public:
  SgmStereo(Config = Config());
  ~SgmStereo();

  inline const Config& config() const { return _config; }
  inline       Config& config()       { return _config; }

  void compute(const cv::Mat&, const cv::Mat&, cv::Mat&);

 private:
  Config _config;
  struct Impl;
  Impl* _impl;
}; // Sgm


#endif // SGM
