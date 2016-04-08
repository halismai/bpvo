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

#include "bpvo/vo.h"
#include "bpvo/timer.h"
#include "bpvo/utils.h"
#include <opencv2/highgui/highgui.hpp>

//
// simple example (without the bpvo_utils)
//

static const char* LEFT_IMAGE_PREFIX =
"/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_%05d.png";

/*
static const char* RIGHT_IMAGE_PREFIX =
"/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/right/tsukuba_fluorescent_R_%05d.png";
*/

static const char* DMAP_PREFIX =
"/home/halismai/data/NewTsukubaStereoDataset/groundtruth/disparity_maps/left/tsukuba_disparity_L_%05d.png";

using namespace bpvo;

int main()
{
  //
  // configure the parameters
  //
  AlgorithmParameters params;
  params.numPyramidLevels = 4;
  params.maxIterations = 100;
  params.parameterTolerance = 1e-6;
  params.functionTolerance = 1e-6;
  params.verbosity = VerbosityType::kSilent;
  params.minTranslationMagToKeyFrame = 0.1;
  params.minRotationMagToKeyFrame = 2.5;
  params.maxFractionOfGoodPointsToKeyFrame = 0.7;
  params.goodPointThreshold = 0.8;

  Matrix33 K; K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;
  float b = 0.1;
  VisualOdometry vo(K, b, ImageSize(480, 640), params);

  double total_time = 0.0f;
  cv::Mat I0, I1, D;
  for(int i = 1; i < 500; ++i) {
    I0 = cv::imread(Format(LEFT_IMAGE_PREFIX, i), cv::IMREAD_GRAYSCALE);
    if(I0.empty()) {
      Warn("Failed to read image %d\n", i);
      break;
    }

    D = cv::imread(Format(DMAP_PREFIX, i), cv::IMREAD_UNCHANGED);
    if(D.empty()) {
      Warn("Failed to read disparity %d\n", i);
      break;
    }

    D.convertTo(D, CV_32FC1);

    Timer timer;
    auto result = vo.addFrame(I0.ptr<uint8_t>(), D.ptr<float>());
    auto tt = timer.stop().count();
    total_time += ( tt / 1000.0f);

    fprintf(stdout, "Frame %03d [%03d ms] %0.2f Hz\r", i, (int) tt, i / total_time);
    fflush(stdout);
  }

  return 0;
}
