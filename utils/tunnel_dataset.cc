#include "utils/tunnel_dataset.h"
#include "bpvo/config_file.h"

#include <fstream>
#include <algorithm>

namespace {

static inline void removeWhiteSpace(std::string& s)
{
  s.erase(std::remove_if(s.begin(), s.end(),
                 [](char c) { return std::isspace<char>(c, std::locale::classic()); }),
          s.end());
}

} // namespace


namespace bpvo {

TunnelDataset::TunnelDataset(std::string conf_fn)
    : DisparityDataset(conf_fn)
{
  ConfigFile cf(conf_fn);
  auto calib_fn = cf.get<std::string>("CalibrationFile");

  THROW_ERROR_IF(!loadCalibration(calib_fn),
                 Format("failed to load calibration from %s\n", calib_fn.c_str())
                 .c_str());
}

TunnelDataset::~TunnelDataset() {}

bool TunnelDataset::loadCalibration(std::string calib_fn)
{
  std::ifstream ifs(calib_fn);
  if(!ifs.is_open())
    return false;
  // THROW_ERROR_IF(!ifs.is_open(), "failed to open calibration file");

  std::string line;
  std::getline(ifs, line); // version
  if(line == "CRL Camera Config") {
    std::getline(ifs, line);
    removeWhiteSpace(line);
    int rows = 0, cols = 0;
    sscanf(line.c_str(), "Width,height:%d,%d", &cols, &rows);
    this->_image_size.rows = rows;
    this->_image_size.cols = cols;
    std::getline(ifs, line); // fps
    std::getline(ifs, line);
    removeWhiteSpace(line);
    float fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    sscanf(line.c_str(), "fx,fy,cx,cy:%f,%f,%f,%f", &fx, &fy, &cx, &cy);
    _calib.K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    std::getline(ifs, line);
    removeWhiteSpace(line);
    sscanf(line.c_str(), "xyzrpq:%f", &_calib.baseline);
    if(_calib.baseline < 0.0f)
      _calib.baseline *= -1.0f;
  } else {
    std::getline(ifs, line); // the camera calibration

    int rows = 0, cols = 0;
    float fx = 0.0, fy = 0.0f, cx = 0.0f, cy = 0.0f;
    float dist_coeffs[6];

    removeWhiteSpace(line);
    sscanf(line.c_str(), "CameraIntrinsicsPlumbBob{%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}",
           &cols, &rows, &fx, &fy, &cx, &cy,
           &dist_coeffs[0], &dist_coeffs[1], &dist_coeffs[2],
           &dist_coeffs[3], &dist_coeffs[4], &dist_coeffs[5]);

    this->_image_size.rows = rows;
    this->_image_size.cols = cols;

    _calib.K <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0;

    std::getline(ifs, line); // CameraIntrinsicsPlumbBob second line
    THROW_ERROR_IF(line.empty(), "malformatted line in calibration file");

    for(int i = 0; i < 4; ++i) {
      std::getline(ifs, line);
      THROW_ERROR_IF(line.empty(), "malformatted line in calibration file");
    }

    std::getline(ifs, line);
    THROW_ERROR_IF(line.empty(), "malformatted line in calibration file");
    float dummy = 0.0f;
    removeWhiteSpace(line);
    sscanf(line.c_str(), "Transform3D(%f,%f,%f,%f",
           &dummy, &dummy, &dummy, &_calib.baseline);

    if(_calib.baseline < 0)
      _calib.baseline *= -1;
  }

  return true;
}

} // bpvo

