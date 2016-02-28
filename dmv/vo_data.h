#ifndef BPVO_DMV_VO_DATA_H
#define BPVO_DMV_VO_DATA_H

#include <bpvo/types.h>
#include <bpvo/utils.h>
#include <bpvo/point_cloud.h>

#include <opencv2/core/core.hpp>

#include <utils/eigen_cereal.h>
#include <utils/cv_cereal.h>

namespace bpvo {
namespace dmv {

class VoDataLive : public VoDataAbstract
{
 public:
  inline VoDataLive(const cv::Mat& I, const PointCloud& pc)
      : _image(I), _point_cloud(pc) {}

  virtual ~VoDataLive() {}

  inline cv::Mat image() const { return _image; }
  inline const PointCloud& pointCloud() const { return _point_cloud; }

  template <class Archive> inline
  void serialize(Archive& ar)
  {
    ar(_image, _point_cloud);
  }

 protected:
  cv::Mat _image;
  PointCloud _point_cloud;
}; // VoDataLive

class VoDataFromDisk : public VoDataAbstract
{
 public:
  VoDataFromDisk(std::string filename, const PointCloud& pc)
      : _filename(filename), _point_cloud(pc) {}

  virtual ~VoDataFromDisk() {}

  cv::Mat image() const;
  inline const PointCloud& pointCloud() const { return _point_cloud;  }

  template <class Archive> inline
  void serialize(Archive& ar)
  {
    ar(_filename, _point_cloud);
  }

 protected:
  std::string _filename;
  PointCloud _point_cloud;
}; // VoDataFromDisk


}; // dmv
}; // bpvo

#endif // BPVO_DMV_VO_DATA_H
