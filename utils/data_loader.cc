#include "utils/data_loader.h"
#include "utils/stereo_algorithm.h"
#include "utils/tunnel_data_loader.h"
#include "utils/glob.h"
#include "bpvo/config_file.h"
#include "bpvo/utils.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

UniquePointer<DataLoader> DataLoader::FromConfig(std::string conf_fn)
{
  ConfigFile cf(conf_fn);
  std::cout << cf << std::endl;
  const auto dataset = cf.get<std::string>("DataSet");
  if(icompare("tsukuba", dataset)) {
    dprintf("TsukubaDataLoader\n");
    return TsukubaDataLoader::Create(cf);
  } else if(icompare("kitti", dataset)) {
    dprintf("KittiDataLoader\n");
    return KittiDataLoader::Create(cf);
  } else if(icompare("tunnel", dataset)) {
    dprintf("tunnel data\n");
    return TunnelDataLoader::Create(cf);
  } else if(icompare("bumblebee", dataset)) {
    return BumblebeeDataLoader::Create(cf);
  } else {
    char buf[1024];
    snprintf(buf, 1024, "Unknown dataset %s\n", dataset.c_str());
    THROW_ERROR(buf);
  }
}

UniquePointer<DataLoader> TsukubaDataLoader::Create(const ConfigFile& cf)
{
  return UniquePointer<DataLoader>(new TsukubaDataLoader(cf));
}

TsukubaDataLoader::TsukubaDataLoader(std::string conf_fn)
  : _root_dir(fs::expand_tilde("~/data/NewTsukubaStereoDataset/")),
    _illumination("fluorescent")
{
  if(!conf_fn.empty()) {
    ConfigFile cf(conf_fn);
    _root_dir = fs::expand_tilde(
        cf.get<std::string>("DataSetRootDirectory", "~/data/NewTsukubaStereoDataset/"));
    _illumination = cf.get<std::string>("Illumination", "fluorescent");
  }

  const auto err_msg = Format("data root directory %s does not exist", _root_dir.c_str());
  THROW_ERROR_IF(!fs::exists(_root_dir), err_msg.c_str());
  THROW_ERROR_IF(_illumination.empty(), "illumination is invalid");
}

TsukubaDataLoader::TsukubaDataLoader(const ConfigFile& cf)
  : _root_dir(fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory")))
   , _illumination(cf.get<std::string>("Illumination"))
{
  const auto err_msg = Format("data root directory %s does not exist", _root_dir.c_str());
  THROW_ERROR_IF(!fs::exists(_root_dir), err_msg.c_str());
  THROW_ERROR_IF(_illumination.empty(), "illumination is invalid");
}

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

  return ImageFramePointer(new StereoFrame(I1, I2, D));
}

StereoDataLoader::StereoDataLoader(const ConfigFile& cf)
  : _stereo_alg(make_unique<StereoAlgorithm>(cf))
  , _left_fmt(fs::expand_tilde(cf.get<std::string>("LeftImageFormat", "")))
  , _right_fmt(fs::expand_tilde(cf.get<std::string>("RightImageFormat", "")))
  , _scale_by(cf.get<int>("ScaleBy", 1))
{
  if(!_left_fmt.empty())
    set_image_size();
}

StereoDataLoader::StereoDataLoader(std::string conf_fn)
  : StereoDataLoader(ConfigFile(conf_fn)) {}

StereoDataLoader::~StereoDataLoader() {}

auto StereoDataLoader::getFrame(int f_i) const -> ImageFramePointer
{
  auto fn = Format(_left_fmt.c_str(), f_i);
  auto I1 = cv::imread(fn, cv::IMREAD_GRAYSCALE);

  auto err_msg = Format("Failed to read image from '%s'", fn.c_str());
  THROW_ERROR_IF(I1.empty(), err_msg.c_str());

  fn = Format(_right_fmt.c_str(), f_i);
  auto I2 = cv::imread(fn, cv::IMREAD_GRAYSCALE);

  err_msg = Format("Failed to read image from '%s'", fn.c_str());
  THROW_ERROR_IF(I2.empty(), err_msg.c_str());

  if(_scale_by > 1) {
    float s = 1.0f / _scale_by;
    cv::resize(I1, I1, cv::Size(), s, s);
    cv::resize(I2, I2, cv::Size(), s, s);
  }

  cv::Mat D;
  _stereo_alg->run(I1, I2, D);

  return ImageFramePointer(new StereoFrame(I1, I2, D));
}

ImageSize StereoDataLoader::imageSize() const { return _image_size; }

void StereoDataLoader::set_image_size()
{
  auto frame = this->getFrame(this->firstFrameNumber());
  this->_image_size = ImageSize(frame->image().rows, frame->image().cols);
}

DisparityDataLoader::DisparityDataLoader(const ConfigFile& cf)
  : _image_format(fs::expand_tilde(cf.get<std::string>("LeftImageFormat", ""))),
    _disparity_format(fs::expand_tilde(cf.get<std::string>("DisparityFormat", "")))
{
  if(!_image_format.empty())
    set_image_size();
}

DisparityDataLoader::DisparityDataLoader(std::string fn)
  : DisparityDataLoader(ConfigFile(fn)) {}

DisparityDataLoader::~DisparityDataLoader() {}

auto DisparityDataLoader::getFrame(int f_i) const -> ImageFramePointer
{
  auto fn = Format(_image_format.c_str(), f_i);
  auto I1 = cv::imread(fn, cv::IMREAD_GRAYSCALE);
  auto err_msg = Format("failed to read image from %s\n", fn.c_str());
  THROW_ERROR_IF(I1.empty(), err_msg.c_str());

  fn = Format(_disparity_format.c_str(), f_i);
  err_msg = Format("failed to read image from %s\n", fn.c_str());
  auto D = cv::imread(fn, cv::IMREAD_UNCHANGED);
  assert( D.type() == cv::DataType<uint16_t>::type && D.channels() == 1 );

  D.convertTo(D, CV_32FC1, 1.0f/16.0, 0.0);

  return ImageFramePointer(new DisparityFrame(I1, D));
}

void DisparityDataLoader::set_image_size()
{
  auto frame = this->getFrame(this->firstFrameNumber());
  this->_image_size = ImageSize(frame->image().rows, frame->image().cols);
}

ImageSize DisparityDataLoader::imageSize() const { return _image_size; }


KittiDataLoader::KittiDataLoader(const ConfigFile& cf)
  : StereoDataLoader(cf)
{
  auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
  THROW_ERROR_IF( !fs::exists(root_dir), "DataSetRootDirectory does not exist" );

  int sequence = cf.get<int>("SequenceNumber");
  THROW_ERROR_IF( sequence < 0 || sequence > 22, "invalid sequence number" );

  this->_left_fmt = Format("%s/sequences/%02d/image_0/%s.png", root_dir.c_str(), sequence, "%06d");
  this->_right_fmt = Format("%s/sequences/%02d/image_1/%s.png", root_dir.c_str(), sequence, "%06d");

  auto calib_fn = Format("%s/sequences/%02d/calib.txt", root_dir.c_str(), sequence);
  THROW_ERROR_IF( !fs::exists(calib_fn), "could not find calib.txt" );
  load_calibration(calib_fn);

  this->set_image_size();
}

KittiDataLoader::KittiDataLoader(std::string conf_fn)
  : KittiDataLoader(ConfigFile(conf_fn)) {  }

KittiDataLoader::~KittiDataLoader() {}

StereoCalibration KittiDataLoader::calibration() const
{
  return _calib;
}

static inline Matrix34 set_kitti_camera_from_line(std::string line)
{
  auto tokens = splitstr(line);
  THROW_ERROR_IF( tokens.empty() || tokens[0].empty() || tokens[0][0] != 'P',
                 "invalid calibration line");
  THROW_ERROR_IF( tokens.size() != 13, "wrong line length" );

  std::vector<float> vals;
  for(size_t i = 1; i < tokens.size(); ++i)
    vals.push_back(str2num<float>(tokens[i]));

  Matrix34 ret;
  for(int r = 0, i = 0; r < ret.rows(); ++r)
    for(int c = 0; c < ret.cols(); ++c, ++i)
      ret(r,c) = vals[i];

  return ret;
}

void KittiDataLoader::load_calibration(std::string filename)
{
  std::ifstream ifs(filename);
  THROW_ERROR_IF( !ifs.is_open(), "failed to open calib.txt" );

  Matrix34 P1, P2;
  std::string line;

  // the first camera
  std::getline(ifs, line);
  P1 = set_kitti_camera_from_line(line);

  std::getline(ifs, line);
  P2 = set_kitti_camera_from_line(line);

  _calib.K = P1.block<3,3>(0,0);
  _calib.baseline =  -P2(0,3) / P2(0,0);

  if(_scale_by > 1) {
    float s = 1.0f / _scale_by;
    _calib.K *= s;
    _calib.K(2,2) = 1.0f;
  }
}

UniquePointer<DataLoader> KittiDataLoader::Create(const ConfigFile& cf)
{
  return UniquePointer<DataLoader>(new KittiDataLoader(cf));
}

BumblebeeDataLoader::BumblebeeDataLoader(const ConfigFile& cf)
  : StereoDataLoader(cf)
{
  auto root = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));

  const auto err_msg = Format("data root directory %s does not exist", root.c_str());
  THROW_ERROR_IF(!fs::exists(root), err_msg.c_str());

  auto left_glob_pattern = root + cf.get<std::string>("LeftGlobPattern", "*left-rect*png");
  auto right_glob_pattern = root + cf.get<std::string>("RightGlobPattern", "*right-rect*png");

  printf("%s\n", left_glob_pattern.c_str());

  _left = glob(left_glob_pattern);
  _right = glob(right_glob_pattern);

  THROW_ERROR_IF( _left.empty(), "could not glob from pattern" );
  THROW_ERROR_IF( _left.size() != _right.size(), "number of files mismatch" );

  // fixed for now
  float f = 487.109;
  float cx = 320.788;
  float cy = 245.845;
  _calib.K << f, 0, cx, 0, f, cy, 0, 0, 1;
  _calib.baseline = 0.120006;

  this->_first_frame_number = cf.get<int>("firstFrameNumber", 0);
}

BumblebeeDataLoader::BumblebeeDataLoader(std::string conf_fn)
  : BumblebeeDataLoader(ConfigFile(conf_fn)) {}

StereoCalibration BumblebeeDataLoader::calibration() const { return _calib; }

BumblebeeDataLoader::~BumblebeeDataLoader() {}

auto BumblebeeDataLoader::getFrame(int f_i) const -> ImageFramePointer
{
  if(f_i >= 0 && f_i < (int) _left.size())
  {
    auto I1 = cv::imread(_left[f_i], cv::IMREAD_GRAYSCALE),
         I2 = cv::imread(_right[f_i], cv::IMREAD_GRAYSCALE);

    auto msg = Format("failed to load %s\n", _left[f_i].c_str());
    THROW_ERROR_IF(I1.empty() || I2.empty(), msg.c_str());

    if(this->_scale_by > 1) {
      float s = 1.0f / _scale_by;
      cv::resize(I1, I1, cv::Size(), s, s);
      cv::resize(I2, I2, cv::Size(), s, s);
    }

    cv::Mat D;
    this->_stereo_alg->run(I1, I2, D);
    return ImageFramePointer(new StereoFrame(I1, I2, D));
  }

  return nullptr;
}

UniquePointer<DataLoader> BumblebeeDataLoader::Create(const ConfigFile& cf)
{
  return  UniquePointer<DataLoader>(new BumblebeeDataLoader(cf));
}

ImageSize BumblebeeDataLoader::imageSize() const { return ImageSize(480, 640); }

DataLoaderThread::DataLoaderThread(UniquePointer<DataLoader> data_loader,
                                   BufferType& buffer)
  : _data_loader(std::move(data_loader)), _buffer(buffer), _thread([=] { this->start(); }) {}

DataLoaderThread::~DataLoaderThread()
{
  stop();
}

void DataLoaderThread::stop(bool empty_buffer)
{
  if(_is_running) {
    _stop_requested = true;

    if(empty_buffer) {
      //
      // we also need to empty the buffer, because the call to push() blocks
      //
      typename BufferType::value_type frame;
      while( _buffer.pop(&frame, 10) )
        ; // nothing here, just pop the frames
    }

    if(_thread.joinable()) {
      _thread.join();
    }
  }
}

bool DataLoaderThread::isRunning() const { return _is_running; }

void DataLoaderThread::start()
{
  typename BufferType::value_type frame;
  int f_i = _data_loader->firstFrameNumber();

  try {
    _is_running = true;
    while( !_stop_requested && (nullptr != (frame=_data_loader->getFrame(f_i++))))
      _buffer.push(std::move(frame));
  } catch(const std::exception& ex) {
    _is_running = false;
    _buffer.push(nullptr);
  }

  _is_running = false;
}

} // bpvo
