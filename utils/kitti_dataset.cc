#include "utils/kitti_dataset.h"
#include "bpvo/config_file.h"
#include "bpvo/debug.h"
#include "bpvo/utils.h"

#include <fstream>
#include <vector>

namespace {

static inline bpvo::Matrix34 set_kitti_camera_from_line(std::string line)
{
  auto tokens = bpvo::splitstr(line);
  THROW_ERROR_IF( tokens.empty() || tokens[0].empty() || tokens[0][0] != 'P',
                 "invalid calibration line");
  THROW_ERROR_IF( tokens.size() != 13, "wrong line length" );

  std::vector<float> vals;
  for(size_t i = 1; i < tokens.size(); ++i)
    vals.push_back(bpvo::str2num<float>(tokens[i]));

  bpvo::Matrix34 ret;
  for(int r = 0, i = 0; r < ret.rows(); ++r)
    for(int c = 0; c < ret.cols(); ++c, ++i)
      ret(r,c) = vals[i];

  return ret;
}

} // namespace

namespace bpvo {

KittiDataset::KittiDataset(std::string conf_fn)
    : StereoDataset(conf_fn)
{
  THROW_ERROR_IF( !this->init(conf_fn), "failed to initialize KittiDataset" );
}

KittiDataset::~KittiDataset(){}

bool KittiDataset::init(const ConfigFile& cf)
{
  try
  {
    auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
    auto sequence = cf.get<int>("SequenceNumber");

    auto left_fmt = Format("sequences/%02d/image_0/%s.png", sequence, "%06d");
    auto right_fmt = Format("sequences/%02d/image_1/%s.png", sequence, "%06d");
    auto frame_start = cf.get<int>("FirstFrameNumber", 0);

    this->_left_filenames = make_unique<FileLoader>(root_dir, left_fmt, frame_start);
    this->_right_filenames = make_unique<FileLoader>(root_dir, right_fmt, frame_start);

    auto frame = this->getFrame(0);
    THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
    this->_image_size = Dataset::GetImageSize(frame.get());

    auto calib_fn = Format("%s/sequences/%02d/calib.txt", root_dir.c_str(), sequence);
    THROW_ERROR_IF(!fs::exists(calib_fn), "calibration file does not exist");
    return loadCalibration(calib_fn);

  } catch(std::exception& ex)
  {
    Warn("Error %s\n", ex.what());
    return false;
  }

  return true;
}

bool KittiDataset::loadCalibration(std::string filename)
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

  if(this->_scale_by > 1) {
    printf("scaling the calibration by %d\n", this->_scale_by);
    float s = 1.0f / _scale_by;
    _calib.K *= s;
    _calib.K(2,2) = 1.0f;
    _calib.baseline /= s;
  }

  return true;
}

}; // bpvo
