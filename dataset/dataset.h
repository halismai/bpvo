#ifndef BPVO_DATASET_DATASET_H
#define BPVO_DATASET_DATASET_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class StereoCalibration;
class ConfigFile;

namespace dataset {

class Frame
{
 public:
  typedef SharedPointer<Frame> Pointer;

 public:
  virtual ~Frame();

  virtual const cv::Mat& image() const = 0;
  virtual const cv::Mat& disparity() const = 0;
}; // Frame

class Dataset
{
 public:
  typedef typename Frame::Pointer FramePointer;

 public:
  enum class Type { Stereo, Disparity, Depth };

 public:
  Dataset(std::string conf_file);
  virtual ~Dataset();

  virtual FramePointer getFrame(int);
  virtual ImageSize imageSize() const;
  virtual const StereoCalibration& calibration() const;
  virtual Dataset::Type type() const;
  virtual std::string name() const;

 protected:
  void loadCalibration(const ConfigFile&);

 private:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // Dataset

}; // dataset

}; // bpvo

#endif // BPVO_DATASET_DATASET_H
