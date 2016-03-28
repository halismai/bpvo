#ifndef RSGM_H
#define RSGM_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

class RSGM
{
 public:
  struct Config
  {
    enum class SubpixelRefinmentMethod {
      None = -1,
      Equiangular,
      Parabolic
    }; // SubpixelRefinmentMethod

    enum class CensusMask
    {
      C5x5,
      C9x7
    }; // CensusMask

    uint16_t P1;
    uint16_t invalidDisparityCost;
    uint16_t numPasses;
    uint16_t numPaths;
    float uniquenessRatio;
    bool doMedianFilter;
    bool doLeftRightCheck;
    bool doRightLeftCheck;
    bool disp12MaxDiff;

    SubpixelRefinmentMethod subpixelRefinmentMethod;

    float alpha;
    uint16_t gamma;
    uint16_t P2min;

    CensusMask censusMask;

    Config()
        : P1(7),
        invalidDisparityCost(12),
        numPasses(2),
        numPaths(2),
        uniquenessRatio(0.95f),
        doMedianFilter(true),
        doLeftRightCheck(true),
        doRightLeftCheck(true),
        disp12MaxDiff(1),
        subpixelRefinmentMethod(SubpixelRefinmentMethod::Parabolic),
        alpha(0.25f),
        gamma(50),
        P2min(17),
        censusMask(CensusMask::C5x5) {}
  }; // Config

 public:
  RSGM(Config = Config());
  ~RSGM();

  void compute(const cv::Mat&, const cv::Mat&, cv::Mat&);

 private:
  Config _config;

  struct Impl;
  bpvo::UniquePointer<Impl> _impl;
}; // RSGM


#endif // RSGM_H
