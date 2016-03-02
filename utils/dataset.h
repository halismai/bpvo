#ifndef BPVO_UTILS_DATASET_H
#define BPVO_UTILS_DATASET_H

#include <bpvo/types.h>

#include <utils/stereo_calibration.h>
#include <utils/file_loader.h>

#include <opencv2/core/core.hpp>

namespace bpvo {

class ConfigFile;

enum class DatasetType
{
  Stereo,    /** data is left and right stereo pairs */
  Disparity, /** data is left image and a disparity */
  Depth      /** RGBD, or image and depth */
}; // DataseType

struct DatasetFrame
{
  virtual const cv::Mat& image() const = 0;
  virtual const cv::Mat& disparity() const = 0;

  virtual std::string filename() const { return ""; }

  virtual ~DatasetFrame() {}
};

class Dataset
{
 public:
  Dataset() {}
  virtual ~Dataset() {}

  /**
   * Get the i-th frame
   */
  virtual UniquePointer<DatasetFrame> getFrame(int f_i) const = 0;

  /**
   * \return the image size
   */
  virtual ImageSize imageSize() const = 0;

  /**
   * \return the calibration
   */
  virtual StereoCalibration calibration() const = 0;

  /**
   * \return the type of the dataset
   */
  virtual DatasetType type() const = 0;

  /**
   * \return the name of the dataset
   */
  virtual std::string name() const = 0;


  /**
   * create a dataset loader from a config file
   */
  static UniquePointer<Dataset> Create(std::string conf_fn);

  /**
   * get the image size from a frame
   */
  static inline ImageSize GetImageSize(const DatasetFrame* f)
  {
    return ImageSize(f->image().rows, f->image().cols);
  }

 protected:
  /** the index of the first frame when doing a printf style image loading */
  int _first_frame_index = 0;
}; // Dataset


class DisparityDataset : public Dataset
{
 public:
  /**
   * frame is composed of the left image and a disparity
   */
  struct DisparityFrame : DatasetFrame
  {
    cv::Mat I_orig; //< the original image
    cv::Mat I;      //< grayscale image
    cv::Mat D;      //< disparity as float
    std::string fn; //< filename if read from disk

    DisparityFrame();
    DisparityFrame(cv::Mat I_, cv::Mat D_, cv::Mat I_orig_ = cv::Mat(),
                   std::string image_filename = "");

    inline const cv::Mat& image() const { return I; }
    inline const cv::Mat& disparity() const { return D; }
    inline std::string filename() const { return fn; }

    virtual ~DisparityFrame() {}
  }; // DisparityFrame

 public:
  DisparityDataset(std::string config_file);
  virtual ~DisparityDataset();

  virtual StereoCalibration calibration() const = 0;
  virtual std::string name() const  = 0;

  inline DatasetType type() const { return DatasetType::Disparity; }
  inline ImageSize imageSize() const { return _image_size; }

  UniquePointer<DatasetFrame> getFrame(int f_i) const;

 protected:
  ImageSize _image_size;
  UniquePointer<FileLoader> _image_filenames;
  UniquePointer<FileLoader> _disparity_filenames;

  bool init(const ConfigFile&);

  float _disparity_scale = 1.0 / 16.0;
}; // DisparityDataset

class StereoAlgorithm;

class StereoDataset : public Dataset
{
 public:
  struct StereoFrame : DatasetFrame
  {
    std::string fn;
    cv::Mat I_orig[2]; //< original stereo images  {left, right}
    cv::Mat I[2];      //< grayscale {left, right}
    cv::Mat D;         //< disparity as float

    inline const cv::Mat& image() const { return I[0]; }
    inline const cv::Mat& disparity() const { return D; }
    inline std::string filename() const { return fn; }

    virtual ~StereoFrame() {}
  }; // StereoFrame

 public:
  StereoDataset(std::string conf_fn);
  virtual ~StereoDataset();

  virtual StereoCalibration calibration() const = 0;
  virtual std::string name() const = 0;

  inline DatasetType type() const { return DatasetType::Stereo; }
  inline ImageSize imageSize() const { return _image_size; }

  UniquePointer<DatasetFrame> getFrame(int f_i) const;

  const StereoAlgorithm* stereo() const;

 protected:
  UniquePointer<StereoAlgorithm> _stereo_alg;
  ImageSize _image_size;

  UniquePointer<FileLoader> _left_filenames;
  UniquePointer<FileLoader> _right_filenames;

  int _scale_by = 1;

  virtual bool init(const ConfigFile&);
}; // StereoDataset

}; // bpvo

#endif // BPVO_UTILS_DATASET_H
