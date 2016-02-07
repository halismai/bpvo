#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <bpvo/types.h>
#include <test/bounded_buffer.h>

#include <iosfwd>
#include <string>
#include <thread>
#include <atomic>

#include <opencv2/core/core.hpp>

namespace bpvo {

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
}; // DataLoader


class TsukubaDataLoader : public DataLoader
{
 public:
  typedef typename DataLoader::ImageFramePointer ImageFramePointer;

 public:
  TsukubaDataLoader(std::string root_dir = "~/data/NewTsukubaStereoDataset/",
                    std::string illumination = "fluorescent");

  virtual ~TsukubaDataLoader();

  StereoCalibration calibration() const;
  ImageFramePointer getFrame(int) const;
  ImageSize imageSize() const;

  inline int firstFrameNumber() const { return 1; }

 private:
  std::string _root_dir;
  std::string _illumination;
}; // TsukubaDataLoader


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
