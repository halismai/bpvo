#include "utils/dataset.h"
#include "utils/stereo_algorithm.h"

#include "bpvo/config_file.h"
#include "bpvo/debug.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

static void toGray(const cv::Mat& src, cv::Mat& ret)
{
  switch( src.type() )
  {
    case CV_8UC1: ret = src; break;
    case CV_8UC3: { cv::cvtColor(src, ret, CV_BGR2GRAY); } break;
    case CV_8UC4: { cv::cvtColor(src, ret, CV_BGRA2GRAY); } break;
    default: THROW_ERROR("unsupported image format");
  }
}

DisparityDataset::DisparityFrame::DisparityFrame() {}

DisparityDataset::DisparityFrame::
DisparityFrame(cv::Mat I_, cv::Mat D_, cv::Mat I_orig_, std::string fn_)
    : I_orig(I_orig_), I(I_), D(D_), fn(fn_) {}

DisparityDataset::DisparityDataset(std::string conf_fn)
{
  ConfigFile cf(conf_fn);
  _disparity_scale = cf.get<float>("DisparityScale", 1.0/16.0);

  THROW_ERROR_IF( !this->init(cf), "failed to initialize DisparityDataset" );
}

DisparityDataset::~DisparityDataset() {}

UniquePointer<DatasetFrame> DisparityDataset::getFrame(int f_i) const
{
  THROW_ERROR_IF( _image_filenames == nullptr, "dataset is not initialized" );

  auto image_fn = _image_filenames->operator[](f_i);
  cv::Mat I = cv::imread(image_fn, cv::IMREAD_UNCHANGED);
  cv::Mat D = cv::imread(_disparity_filenames->operator[](f_i), cv::IMREAD_UNCHANGED);

  if(I.empty() || D.empty())
    return nullptr;

  THROW_ERROR_IF( D.channels() > 1, "disparity must be a single channel" );
  THROW_ERROR_IF( I.size() != D.size(), "frame size mismatch" );

  if(D.type() != cv::DataType<float>::type)
    D.convertTo(D, CV_32FC1, _disparity_scale, 0.0);

  cv::Mat I_gray;
  toGray(I, I_gray);
  return UniquePointer<DatasetFrame>(new DisparityFrame(I_gray, D, I, image_fn));
}

bool DisparityDataset::init(const ConfigFile& cf)
{
  try
  {
    auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
    THROW_ERROR_IF( !fs::exists(root_dir),
                   Format("DataSetRootDirectory '%s' does not exist", root_dir.c_str()).c_str() );

    auto left_fmt = cf.get<std::string>("LeftImageFormat", "");
    auto dmap_fmt = cf.get<std::string>("DisparityMapFormat", "");
    auto frame_start = cf.get<int>("FirstFrameNumber", 0);

    //
    // allow children to set this later
    //
    if(!left_fmt.empty())
    {
      _image_filenames = make_unique<FileLoader>(root_dir, left_fmt, frame_start);
      _disparity_filenames = make_unique<FileLoader>(root_dir, dmap_fmt, frame_start);

      auto frame = this->getFrame(0);
      THROW_ERROR_IF( frame == nullptr, "failed to read frame" );
      _image_size = Dataset::GetImageSize(frame.get());
    }
  } catch(const std::exception& ex)
  {
    Warn("Error %s\n", ex.what());
    return false;
  }

  return true;
}

StereoDataset::StereoDataset(std::string conf_fn)
: _stereo_alg(new StereoAlgorithm(ConfigFile(conf_fn)))
    , _scale_by( ConfigFile(conf_fn).get<int>("ScaleBy", 1) )
{
  THROW_ERROR_IF( !this->init(conf_fn), "failed to initialize StereoDataset" );
}

StereoDataset::~StereoDataset() {}

UniquePointer<DatasetFrame> StereoDataset::getFrame(int f_i) const
{
  THROW_ERROR_IF( _left_filenames == nullptr || _right_filenames == nullptr,
                 "has not been initialized" );

  auto image_fn = _left_filenames->operator[](f_i);
  StereoFrame frame;
  frame.I_orig[0] = cv::imread(image_fn, cv::IMREAD_UNCHANGED);
  frame.I_orig[1] = cv::imread(_right_filenames->operator[](f_i), cv::IMREAD_UNCHANGED);

  if(frame.I_orig[0].empty() || frame.I_orig[1].empty())
  {
    dprintf("nore more images?\nleft:%s\nright:%s",
            _left_filenames->operator[](f_i).c_str(),
            _right_filenames->operator[](f_i).c_str());

    return nullptr;
  }

  toGray(frame.I_orig[0], frame.I[0]);
  toGray(frame.I_orig[1], frame.I[1]);

  if(_scale_by > 1) {
    double s = 1.0 / _scale_by;
    cv::resize(frame.I[0], frame.I[0], cv::Size(), s, s);
    cv::resize(frame.I[1], frame.I[1], cv::Size(), s, s);
  }

  frame.fn = image_fn;

  _stereo_alg->run(frame.I[0], frame.I[1], frame.D);
  return UniquePointer<DatasetFrame>(new StereoFrame(frame));
}

const StereoAlgorithm* StereoDataset::stereo() const { return _stereo_alg.get(); }

bool StereoDataset::init(const ConfigFile& cf)
{
  try
  {
    auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
    THROW_ERROR_IF(!fs::exists(root_dir), "DataSetRootDirectory does not exist");

    //
    // we'll allow subclasses to do this part
    //
    auto left_fmt = cf.get<std::string>("LeftImageFormat", "");
    auto right_fmt = cf.get<std::string>("RightImageFormat", "");
    auto frame_start = cf.get<int>("FirstFrameNumber", 0);

    if(!left_fmt.empty())
    {
      _left_filenames = make_unique<FileLoader>(root_dir, left_fmt, frame_start);
      _right_filenames = make_unique<FileLoader>(root_dir, right_fmt, frame_start);

      auto frame = this->getFrame(0);
      THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
      _image_size = Dataset::GetImageSize(frame.get());
    }
  } catch(const std::exception& ex)
  {
    Warn("Error %s\n", ex.what());
    return false;
  }

  return true;
}

} // bpvo

