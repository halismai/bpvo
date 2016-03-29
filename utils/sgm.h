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

// this code will be enabled if WITH_GPL_CODE is defined
#ifndef SGM_H
#define SGM_H

namespace cv {
class Mat;
}; // cv

class SgmStereo
{
 public:
  struct Config
  {
    int numberOfDisparities;
    int sobelCapValue;
    int censusRadius;
    int windowRadius;
    int smoothnessPenaltySmall;
    int smoothnessPenaltyLarge;
    int consistencyThreshold;

    double disparityFactor;
    double censusWeightFactor;

    Config();
  }; // Config

 public:
  SgmStereo(Config = Config());
  ~SgmStereo();

  inline const Config& config() const { return _config; }
  inline       Config& config()       { return _config; }

  void compute(const cv::Mat&, const cv::Mat&, cv::Mat&);

 private:
  Config _config;
  struct Impl;
  Impl* _impl;
}; // Sgm


#endif // SGM
