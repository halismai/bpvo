#ifndef BPVO_DMV_PHOTO_VO_PATCH_H
#define BPVO_DMV_PHOTO_VO_PATCH_H

#include <dmv/photo_vo_base.h>
#include <SmallVector.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {
namespace dmv {

/**
 * Pixel based
 */
class PhotoVoPatch : public PhotoVoBase
{
 public:
  /**
   * Apply this scaling to image intensities
   */
  static constexpr double PixelScale = 1.0 / 255.0;

  using PhotoVoBase::Config;
  using PhotoVoBase::Result;

 public:
  /**
   */
  PhotoVoPatch(const Mat_<double,3,3>& K, double b, Config);

  virtual ~PhotoVoPatch();

  Result estimatePose(const cv::Mat& image, const Mat_<double,4,4>& T_init =
                      Mat_<double,4,4>::Identity());


 private:
  typename AlignedVector<double>::type _I0;

 protected:
  virtual void setImageData(const cv::Mat&, const cv::Mat&);

 protected:
  typedef llvm::SmallVector<double, 16> PatchType;
  std::vector<PatchType> _patch_data;
}; // PhotoVo

}; // dmv
}; // bpvo


#endif // BPVO_DMV_PHOTO_VO_PATCH_H
