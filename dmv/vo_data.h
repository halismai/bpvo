#ifndef BPVO_DMV_VO_DATA_H
#define BPVO_DMV_VO_DATA_H

#include <bpvo/types.h>
#include <bpvo/utils.h>
#include <bpvo/point_cloud.h>

#include <opencv2/core/core.hpp>
#include <vector>

#include <utils/eigen_cereal.h>
#include <utils/cv_cereal.h>

namespace bpvo {
namespace dmv {

/**
 * the raw data we get from vo
 */
class VoData
{
 public:
  inline VoData(const cv::Mat& I, const PointCloud& pc)
      : _image(I), _point_cloud(pc) {}

  template <class Archive> inline
  void serialize(Archive& ar)
  {
    ar(_image);
    ar(_point_cloud);
  }

 protected:
  cv::Mat _image;
  PointCloud _point_cloud;
}; // VoData

}; // dmv
}; // bpvo

#endif // BPVO_DMV_VO_DATA_H
