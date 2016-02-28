#ifndef BPVO_VO_OUTPUT_H
#define BPVO_VO_OUTPUT_H

#include <bpvo/point_cloud.h>
#include <bpvo/utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#if defined(WITH_CEREAL)
#include <utils/eigen_cereal.h>
#include <utils/cv_cereal.h>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>
#endif // WITH_CEREAL

namespace bpvo {

/**
 * stores the data we get from VO
 */
class VoOutput
{
 public:
  inline VoOutput() {}

  inline VoOutput(const PointCloud& pc)
      : _point_cloud(pc) {}

  virtual ~VoOutput() {}

  /**
   * The image has special handling
   */
  virtual cv::Mat image() const = 0;

  /**
   */
  virtual const PointCloud& pointCloud() const { return _point_cloud; }

#if defined(WITH_CEREAL)
  template <class Archive> inline
  void serialize(Archive& ar) { ar(_point_cloud); }
#endif

 protected:
  PointCloud _point_cloud;
}; // VoDataAbstract

/**
 * store the live output
 */
class VoOutputLive : public VoOutput
{
 public:
  inline VoOutputLive() {}

  inline VoOutputLive(const PointCloud& pc, const cv::Mat& image)
      : VoOutput(pc), _image(image) {}

  virtual ~VoOutputLive() {}

  virtual cv::Mat image() const { return _image; }

#if defined(WITH_CEREAL)
  template <class Archive> inline
  void serialize(Archive& ar)
  {
    ar(cereal::base_class<VoOutput>(this), _image);
  }
#endif

 protected:
  cv::Mat _image;
}; // VoOutputLive

/**
 * loads the image from disk when requested
 */
class VoOutputFromDisk : public VoOutput
{
 public:
  inline VoOutputFromDisk() {}

  inline VoOutputFromDisk(const PointCloud& pc, const std::string& filename)
      : VoOutput(pc), _filename(filename) {}

  virtual ~VoOutputFromDisk() {}

  virtual cv::Mat image() const
  {
    cv::Mat ret = cv::imread(_filename, cv::IMREAD_GRAYSCALE);
    THROW_ERROR_IF( ret.empty(), Format_("failed to read %s\n", _filename.c_str()));
    return ret;
  }

#if defined(WITH_CEREAL)
  template <class Archive> inline
  void serialize(Archive& ar)
  {
    ar(cereal::base_class<VoOutput>(this), _filename);
  }
#endif

 protected:
  std::string _filename;
}; // VoOutputFromDisk

}; // bpvo

#if defined(WITH_CEREAL)
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
//CEREAL_REGISTER_TYPE(bpvo::VoOutput);
CEREAL_REGISTER_TYPE(bpvo::VoOutputLive);
CEREAL_REGISTER_TYPE(bpvo::VoOutputFromDisk);
#endif

#endif // BPVO_VO_DATA_H

