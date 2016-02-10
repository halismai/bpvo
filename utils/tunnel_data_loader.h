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

#ifndef BPVO_UTILS_TUNNEL_DATA_LOADER_H
#define BPVO_UTILS_TUNNEL_DATA_LOADER_H

#include <utils/data_loader.h>

namespace bpvo {

class TunnelDataLoader : public DisparityDataLoader
{
 public:
  typedef typename DisparityDataLoader::ImageFramePointer ImageFramePointer;

 public:
  TunnelDataLoader(const ConfigFile& cf);

  TunnelDataLoader(std::string);

  virtual ~TunnelDataLoader();

  StereoCalibration calibration() const;

  static UniquePointer<DataLoader> Create(const ConfigFile&);

 private:
  StereoCalibration _calib;
  void load_calibration(std::string filename);
}; // TunnelDataLoader

}; // bpvo

#endif // BPVO_UTILS_TUNNEL_DATA_LOADER_H
