#ifndef BPVO_UTILS_DATASET2_H
#define BPVO_UTILS_DATASET2_H

#include <bpvo/types.h>
#include <utils/stereo_calibration.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {


namespace utils {

class DatasetFrame
{
 public:
  virtual ~DatasetFrame();

  virtual const cv::Mat& image() const = 0;
  virtual const cv::Mat& disparity() const = 0;

  virtual std::string filename() const;

 protected:
  std::string _filename;
}; // DatasetFrame

class Dataset
{
 public:
  enum class Type { Stereo, Disparity, Depth }; // Type

 public:
  virtual ~Dataset();

  virtual UniquePointer<DatasetFrame> getFrame(int);
  virtual ImageSize imageSize() const;
  virtual StereoCalibration calibration() const;
  virtual Dataset::Type type() const;
  virtual std::string name() const;

  static UniquePointer<Dataset> Create(std::string conf_fn);

 protected:
  class Impl;
  UniquePointer<Impl> _impl;
}; // Dataset


}; // utils
}; // bpvo


#endif // BPVO_UTILS_DATASET2_H
