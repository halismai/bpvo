#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <bpvo/types.h>
#include <utils/bounded_buffer.h>

#include <iosfwd>
#include <string>
#include <thread>
#include <atomic>

#include <opencv2/core/core.hpp>

namespace bpvo {

class ConfigFile;

cv::Mat colorizeDisparity(const cv::Mat&);

struct ImageFrame
{
  virtual const cv::Mat& image() const = 0;
  virtual const cv::Mat& disparity() const = 0;

  virtual ~ImageFrame() {}
}; // ImageFrame

class StereoFrame : public ImageFrame
{
 public:
  StereoFrame();
  explicit StereoFrame(const cv::Mat& left, const cv::Mat& right);
  explicit StereoFrame(const cv::Mat& left, const cv::Mat& right, const cv::Mat& dmap);
  virtual ~StereoFrame();

  const cv::Mat& image() const;
  const cv::Mat& disparity() const;

  void setLeft(const cv::Mat&);
  void setRight(const cv::Mat&);
  void setDisparity(const cv::Mat&);

 private:
  cv::Mat _left;
  cv::Mat _right;
  cv::Mat _disparity;
}; // StereoFrame

class DisparityFrame : public ImageFrame
{
 public:
  DisparityFrame();
  explicit DisparityFrame(const cv::Mat& left, const cv::Mat& disparity);
  virtual ~DisparityFrame();

  const cv::Mat& image() const;
  const cv::Mat& disparity() const;

  void setImage(const cv::Mat&);
  void setDisparity(const cv::Mat&);

 private:
  cv::Mat _image;
  cv::Mat _disparity;

 protected:
  void convertDisparityToFloat();
}; // DisparityFrame

struct StereoCalibration
{
  inline StereoCalibration(const Matrix33& K_ = Matrix33::Identity(),
                           float baseline_ = 1.0f)
      : K(K_), baseline(baseline_) {}

  Matrix33 K;
  float baseline;

  friend std::ostream& operator<<(std::ostream&, const StereoCalibration&);
}; // StereoCalibration


struct DataLoader
{
  typedef SharedPointer<ImageFrame> ImageFramePointer;

  virtual StereoCalibration calibration() const = 0;
  virtual ImageFramePointer getFrame(int f_i) const = 0;
  virtual ImageSize imageSize() const = 0;
  virtual int firstFrameNumber() const { return 0; }
  virtual void setFirstFrameNumber(int) {}


  /**
   * \return a data loader from a config file
   */
  static UniquePointer<DataLoader> FromConfig(std::string);
}; // DataLoader


class StereoAlgorithm;

class TsukubaDataLoader : public DataLoader
{
 public:
  typedef typename DataLoader::ImageFramePointer ImageFramePointer;

 public:
  TsukubaDataLoader(const ConfigFile& cf);
  TsukubaDataLoader(std::string config_file_name = "");
  virtual ~TsukubaDataLoader();

  StereoCalibration calibration() const;
  ImageFramePointer getFrame(int) const;
  ImageSize imageSize() const;

  inline int firstFrameNumber() const { return 1; }

  static UniquePointer<DataLoader> Create(const ConfigFile&);

 private:
  std::string _root_dir;
  std::string _illumination;
}; // TsukubaDataLoader


class StereoDataLoader : public DataLoader
{
 public:
  typedef typename DataLoader::ImageFramePointer ImageFramePointer;

 public:
  StereoDataLoader(const ConfigFile& cf);
  StereoDataLoader(std::string conf_fn);
  virtual ~StereoDataLoader();

  virtual StereoCalibration calibration() const = 0;

  ImageSize imageSize() const;
  ImageFramePointer getFrame(int) const;

  inline int firstFrameNumber() const { return 0; }

 private:
  UniquePointer<StereoAlgorithm> _stereo_alg;

 protected:
  std::string _left_fmt;
  std::string _right_fmt;
  ImageSize _image_size;

  void set_image_size();
}; // StereoDataLoader


class KittiDataLoader : public StereoDataLoader
{
 public:
  typedef typename StereoDataLoader::ImageFramePointer ImageFramePointer;

 public:
  explicit KittiDataLoader(const ConfigFile& cf);
  explicit KittiDataLoader(std::string conf_fn);

  virtual ~KittiDataLoader();

  StereoCalibration calibration() const;

  static UniquePointer<DataLoader> Create(const ConfigFile&);

 private:
  StereoCalibration _calib;
  void load_calibration(std::string filename);
}; // KittiDataLoader


class DisparityDataLoader : public DataLoader
{
 public:
  typedef typename DataLoader::ImageFramePointer ImageFramePointer;

 public:
  DisparityDataLoader(const ConfigFile& cf);
  DisparityDataLoader(std::string conf_fn);

  virtual ~DisparityDataLoader();

  virtual StereoCalibration calibration() const = 0;

  ImageSize imageSize() const;
  ImageFramePointer getFrame(int) const;

  inline int firstFrameNumber() const { return 0; }

 protected:
  std::string _image_format;
  std::string _disparity_format;

  ImageSize _image_size;
  void set_image_size();
}; // DisparityDataLoader


//
// starts a data loaders in its own thread
//
class DataLoaderThread
{
 public:
  typedef BoundedBuffer<typename DataLoader::ImageFramePointer> BufferType;

 public:
  DataLoaderThread(UniquePointer<DataLoader> data_loader, BufferType& buffer);
  ~DataLoaderThread();

  void stop();

  bool isRunning() const;

 protected:
  UniquePointer<DataLoader> _data_loader;
  BufferType& _buffer;

  void start();

  std::atomic<bool> _stop_requested{false};
  std::atomic<bool> _is_running{false};

  std::thread _thread;
}; // DataLoaderThread

}; // bpvo

#endif // DATA_LOADER_H
