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

  /**
   * the image as grayscale
   */
  virtual const cv::Mat& image() const = 0;

  /**
   * disparity as float (and scaled properly)
   */
  virtual const cv::Mat& disparity() const = 0;

  /**
   * filename of the image, in case it is loaded from disk
   */
  virtual std::string filename() const;

 protected:
  std::string _filename;
}; // DatasetFrame

class Dataset
{
 public:
  /**
   * Type of the dataset, each type has a slightly different handling
   */
  enum class Type { Stereo, Disparity, Depth }; // Type

 public:
  virtual ~Dataset();

  /**
   * \return the frame at index 'i'
   */
  virtual UniquePointer<DatasetFrame> getFrame(int);

  /**
   * \return the image size
   */
  virtual ImageSize imageSize() const;

  /**
   * \return the calibration
   */
  virtual StereoCalibration calibration() const;

  /**
   * \return the type of the dataset
   */
  virtual Dataset::Type type() const;

  /**
   * \return the name of the dataset
   */
  virtual std::string name() const;

  /**
   * Creates a dataset from a config file
   */
  static UniquePointer<Dataset> Create(std::string conf_fn);

 protected:
  class Impl;
  UniquePointer<Impl> _impl;
}; // Dataset


}; // utils
}; // bpvo


#endif // BPVO_UTILS_DATASET2_H
