/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#include "bpvo/config_file.h"
#include "utils/tunnel_data_loader.h"

#include <algorithm>
#include <fstream>
#include <cctype>

namespace bpvo {

TunnelDataLoader::TunnelDataLoader(const ConfigFile& cf)
    : DisparityDataLoader(cf)
{
  auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
  auto err_msg = Format("directory %s does not exist", root_dir.c_str());
  THROW_ERROR_IF(!fs::exists(root_dir), err_msg.c_str());

  this->_image_format = Format("%s/image%s.pgm", root_dir.c_str(), "%06d");
  this->_disparity_format = Format("%s/image%s-disparity.pgm", root_dir.c_str(), "%06d");

  auto calib_fn = fs::expand_tilde(cf.get<std::string>("CalibrationFile"));
  err_msg = Format("calibration file %s does not exist", calib_fn.c_str());
  THROW_ERROR_IF(!fs::exists(calib_fn), err_msg.c_str());

  load_calibration(calib_fn);

  int first_frame_num = cf.get<int>("firstFrameNumber", 0);
  this->setFirstFrameNumber(first_frame_num);
}

TunnelDataLoader::TunnelDataLoader(std::string conf_fn)
  : TunnelDataLoader(ConfigFile(conf_fn)) {}

TunnelDataLoader::~TunnelDataLoader() {}

static inline void removeWhiteSpace(std::string& s)
{
  s.erase(std::remove_if(s.begin(), s.end(),
                 [](char c) { return std::isspace<char>(c, std::locale::classic()); }),
                 s.end());
}

void TunnelDataLoader::load_calibration(std::string filename)
{
  std::ifstream ifs(filename);
  THROW_ERROR_IF(!ifs.is_open(), "failed to open calibration file");

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
           &rows, &cols, &fx, &fy, &cx, &cy,
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
}

StereoCalibration TunnelDataLoader::calibration() const { return _calib; }

UniquePointer<DataLoader> TunnelDataLoader::Create(const ConfigFile& cf)
{
  return UniquePointer<DataLoader>(new TunnelDataLoader(cf));
}

} // bpvo

