#include "utils/dataset.h"
#include "utils/file_loader.h"
#include "utils/stereo_algorithm.h"
#include "utils/tsukuba_dataset.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"
#include "bpvo/config_file.h"

namespace bpvo {

static StereoCalibration TsukubaCalibration(float scale = 1.0f)
{
  StereoCalibration calib;

  calib.K <<
      615.0, 0.0, 320.0,
      0.0, 615.0, 240.0,
      0.0, 0.0, 1.0;
  calib.baseline = 0.1;

  calib.K *= scale; calib.K(2,2) = 1.0f;
  calib.baseline *= (1.0f / scale);

  return calib;
}

TsukubaSyntheticDataset::TsukubaSyntheticDataset(std::string conf_fn)
    : DisparityDataset(conf_fn), _calib(TsukubaCalibration())
{
  THROW_ERROR_IF( !init(ConfigFile(conf_fn)), "failed to init TsukubaSyntheticDataset" );
}

TsukubaSyntheticDataset::~TsukubaSyntheticDataset() {}

bool TsukubaSyntheticDataset::init(const ConfigFile& cf)
{
  try
  {
    auto root_dir = fs::expand_tilde(
        cf.get<std::string>("DataSetRootDirectory", "~/home/data/NewTsukubaStereoDataset"));
    THROW_ERROR_IF( !fs::exists(root_dir), "DataSetRootDirectory does not exist" );

    auto illumination = cf.get<std::string>("Illumination", "fluorescent");
    auto frame_start = cf.get<int>("FirstFrameNumber", 1); // tsukuba starts from 1

    auto img_fmt = Format("illumination/%s/left/tsukuba_%s_L_%s.png",
                          illumination.c_str(), illumination.c_str(), "%05d");
    auto dmap_fmt = Format("groundtruth/disparity_maps/left/tsukuba_disparity_L_%s.png", "%05d");

    this->_image_filenames = make_unique<FileLoader>(root_dir, img_fmt, frame_start);
    this->_disparity_filenames = make_unique<FileLoader>(root_dir, dmap_fmt, frame_start);

    auto frame = this->getFrame(0);
    THROW_ERROR_IF( nullptr == frame, "failed to load the first frame" );
    _image_size = Dataset::GetImageSize( frame.get() );
  } catch(const std::exception& ex)
  {
    Warn("Error %s\n", ex.what());
    return false;
  }

  return true;
}

TsukubaStereoDataset::TsukubaStereoDataset(std::string conf_fn)
  : StereoDataset(conf_fn), _calib(TsukubaCalibration())
{
  THROW_ERROR_IF( !init(ConfigFile(conf_fn)), "failed to init TsukubaStereoDataset" );
}

TsukubaStereoDataset::~TsukubaStereoDataset() {}

bool TsukubaStereoDataset::init(const ConfigFile& cf)
{
  try {
    auto root_dir = fs::expand_tilde(
        cf.get<std::string>("DataSetRootDirectory", "~/home/data/NewTsukubaStereoDataset"));
    THROW_ERROR_IF( !fs::exists(root_dir), "DataSetRootDirectory does not exist" );

    auto illumination = cf.get<std::string>("Illumination", "fluorescent");
    auto frame_start = cf.get<int>("FirstFrameNumber", 1); // tsukuba starts from 1

    auto left_fmt = Format("illumination/%s/left/tsukuba_%s_L_%s.png",
                           illumination.c_str(), illumination.c_str(), "%05d");
    auto right_fmt = Format("illumination/%s/right/tsukuba_%s_R_%s.png",
                           illumination.c_str(), illumination.c_str(), "%05d");

    this->_left_filenames = make_unique<FileLoader>(root_dir,  left_fmt, frame_start);
    this->_right_filenames = make_unique<FileLoader>(root_dir, right_fmt, frame_start);

    auto frame = this->getFrame(0);
    THROW_ERROR_IF( nullptr == frame, "Failed to load the first frame" );
    _image_size = Dataset::GetImageSize(frame.get());
  } catch(const std::exception& ex)
  {
    Warn("Error %s\n", ex.what());
    return false;
  }
  return true;
}

} // bpvo
