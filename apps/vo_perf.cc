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

#include "utils/bounded_buffer.h"
#include "utils/program_options.h"
#include "utils/dataset.h"

#include "bpvo/config.h"
#include "bpvo/trajectory.h"
#include "bpvo/utils.h"
#include "bpvo/vo.h"
#include "bpvo/timer.h"

#include <opencv2/highgui/highgui.hpp>

#include <fstream>

using namespace bpvo;

int main(int argc, char** argv)
{
  ProgramOptions options("vo_perf");
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba_stereo.cfg", "config file")
      ("output,o", "", "prefix to store results for later analysis")
      ("numframes,n", int(1000), "number of frames to process")
      ("dontshow,x", "do not show the image").parse(argc, argv);

  const auto conf_fn = options.get<std::string>("config");
  const auto max_frames = options.get<int>("numframes");
  const auto do_show = !options.hasOption("dontshow");
  const auto output_fn = options.get<std::string>("output");

  auto dataset = Dataset::Create(conf_fn);

  AlgorithmParameters params(conf_fn);
  auto maxTestLevel = params.maxTestLevel;
  auto vo = VisualOdometry(dataset.get(), params);

  Trajectory trajectory;
  UniquePointer<DatasetFrame> frame;

  std::vector<int> iterations;
  std::vector<float> time_ms;

  iterations.reserve(max_frames);
  time_ms.reserve(max_frames);

  double total_time = 0.0;
  int f_i;
  for(f_i = 0; f_i < max_frames; ++f_i)
  {
    frame = dataset->getFrame(f_i);
    if(!frame) {
      Info("no more data\n");
      break;
    }

    if(do_show) {
      cv::imshow("image", frame->image());
      int k = 0xff & cv::waitKey(5);
      if('q' == k)
        break;
    }

    Timer timer;
    auto result = vo.addFrame(frame.get());
    double tt = timer.stop().count();
    total_time += (tt / 1000.0);

    int num_iters = result.optimizerStatistics[maxTestLevel].numIterations;
    if(num_iters == params.maxIterations) {
      fprintf(stdout, "\n");
      Warn("max iterations reached at frame %d\n", f_i);
    }

    fprintf(stdout, "Frame %05d %*.2f ms @ %*.2f Hz %03d iters %20s num_points %-*d\r",
              f_i-1, 6, tt, 5, (f_i - 1) / total_time,  num_iters,
              ToString(result.keyFramingReason).c_str(), 8, 0/*vo.numPointsAtLevel()*/);
    fflush(stdout);

    trajectory.push_back(result.pose);
    time_ms.push_back(tt);
    iterations.push_back(num_iters);
  }

  fprintf(stdout, "\n");
  Info("done\n");

  if(!output_fn.empty()) {
    printf("writing results to prefix %s\n", output_fn.c_str());

    {
      trajectory.writeCameraPath( output_fn + "_path.txt" );

      std::ofstream ofs(output_fn + "_poses.txt");
      if(ofs.is_open()) {
        for(size_t i = 0; i < trajectory.size(); ++i)
          ofs << trajectory[i] << "\n";
      }

      ofs.close();
    }

    {
      std::ofstream ofs(output_fn + "_iterations.txt");
      if(ofs.is_open()) {
        for(size_t i = 0; i < iterations.size(); ++i)
          ofs << iterations[i] << "\n";
      }
      ofs.close();
    }

    {
      std::ofstream ofs(output_fn + "_time.txt");
      if(ofs.is_open()) {
        for(size_t i = 0; i < time_ms.size(); ++i)
          ofs << time_ms[i] << "\n";
      }
      ofs.close();
    }
  }

  return 0;
}

