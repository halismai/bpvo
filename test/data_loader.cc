#include "test/data_loader.h"
#include "bpvo/utils.h"
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include<opencv2/contrib/contrib.hpp>

namespace bpvo {

std::ostream& operator<<(std::ostream& os, const StereoCalibration& c)
{
  os << c.K << "\n";
  os << "baseline: " << c.baseline;

  return os;
}

StereoFrame::StereoFrame() {}
StereoFrame::~StereoFrame() {}
StereoFrame::StereoFrame(const cv::Mat& left, const cv::Mat& right)
    : _left(left), _right(right) {}
StereoFrame::StereoFrame(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity)
    : _left(left), _right(right), _disparity(disparity) {}


const cv::Mat& StereoFrame::image() const { return _left; }
const cv::Mat& StereoFrame::disparity() const { return _disparity;  }

void StereoFrame::setLeft(const cv::Mat& I) { _left = I; }
void StereoFrame::setRight(const cv::Mat& I) { _right = I; }
void StereoFrame::setDisparity(const cv::Mat& D) { _disparity = D; }

TsukubaDataLoader::TsukubaDataLoader(std::string root_dir, std::string illumination)
  : _root_dir(fs::expand_tilde(root_dir)), _illumination(illumination) {}

TsukubaDataLoader::~TsukubaDataLoader() {}

StereoCalibration TsukubaDataLoader::calibration() const
{
  Matrix33 K;
  K << 615.0, 0.0, 320.0,
       0.0, 615.0, 240.0,
       0.0, 0.0, 1.0;

  return StereoCalibration(K, 0.1);
}

ImageSize TsukubaDataLoader::imageSize() const { return {480,640}; }

auto TsukubaDataLoader::getFrame(int f_i) const -> ImageFramePointer
{
  if(f_i < 1 || f_i > 1800) {
    return nullptr;
  }

  auto fn = Format("%s/illumination/%s/left/tsukuba_%s_L_%05d.png",
                   _root_dir.c_str(), _illumination.c_str(), _illumination.c_str(), f_i);
  auto I1 = cv::imread(fn, cv::IMREAD_GRAYSCALE);
  THROW_ERROR_IF(I1.empty(), "could not read image from");


  fn = Format("%s/illumination/%s/right/tsukuba_%s_R_%05d.png",
              _root_dir.c_str(), _illumination.c_str(), _illumination.c_str(), f_i);
  auto I2 = cv::imread(fn, cv::IMREAD_GRAYSCALE);
  THROW_ERROR_IF(I2.empty(), "could not read image from");

  fn = Format("%s/groundtruth/disparity_maps/left/tsukuba_disparity_L_%05d.png",
              _root_dir.c_str(), f_i);
  auto D = cv::imread(fn, cv::IMREAD_UNCHANGED);
  THROW_ERROR_IF(D.empty(), "could not read image from");
  D.convertTo(D, CV_32FC1);

  return SharedPointer<ImageFrame>(new StereoFrame(I1, I2, D));
}


cv::Mat colorizeDisparity(const cv::Mat& D)
{
  cv::Mat ret(D);
  double min_val = 0,  max_val = 0;
  cv::minMaxLoc(ret, &min_val, &max_val);

  ret = 255.0 * ((ret - min_val) / (max_val - min_val));
  ret.convertTo(ret, CV_8U);
  cv::applyColorMap(ret, ret, cv::COLORMAP_JET);
  return ret;
}

DataLoaderThread::DataLoaderThread(UniquePointer<DataLoader> data_loader,
                                   BufferType& buffer)
  : _data_loader(std::move(data_loader)), _buffer(buffer), _thread([=] { this->start(); }) {}

DataLoaderThread::~DataLoaderThread() { stop(); }

void DataLoaderThread::stop()
{
  _stop_requested = true;
  if(_thread.joinable())
    _thread.join();
}

bool DataLoaderThread::isRunning() const { return _is_running; }

void DataLoaderThread::start()
{
  typename BufferType::value_type frame;
  int f_i = _data_loader->firstFrameNumber();

  _is_running = true;
  while( !_stop_requested && (nullptr != (frame=_data_loader->getFrame(f_i++))))
  {
    _buffer.push(std::move(frame));
  }

  _is_running = false;
}

} // bpvo
