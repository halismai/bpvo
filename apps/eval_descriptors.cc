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
#include <iostream>

using namespace bpvo;

template <class Container> static inline
void WriteContainer(std::string filename, const Container& c)
{
  std::ofstream ofs(filename);
  if(ofs.is_open()) {
    for(const auto& c_i : c)
      ofs << c_i << "\n";
  } else {
    Warn("Failed to open file '%s'\n", filename.c_str());
  }
}


static void Run(std::string conf_fn, std::string prefix, int numframes, std::string desc_name)
{
  Info("Running %s\n", desc_name.c_str());

  AlgorithmParameters params(conf_fn);
  params.descriptor = DescriptorTypeFromString(desc_name);

  auto dataset = Dataset::Create(conf_fn);
  auto vo = VisualOdometry(dataset.get(), params);

  Trajectory trajectory;
  std::vector<double> time_ms;
  std::vector<int> num_iterations;

  time_ms.reserve(numframes);
  num_iterations.reserve(numframes);

  double total_time = 0.0;

  UniquePointer<DatasetFrame> frame;
  for(int f_i = 0; f_i < numframes; ++f_i) {
    frame = dataset->getFrame(f_i);

    if(!frame) {
      Warn("could not get frame %d\n", f_i);
      break;
    }

    cv::imshow("image", frame->image());
    int k = cv::waitKey(5) & 0xff;
    if(k == 'q') break;

    Timer timer;
    auto result = vo.addFrame(frame.get());
    double tt = timer.stop().count();
    total_time += (tt / 1000.0);

    int num_iters = result.optimizerStatistics.front().numIterations;
    if(num_iters == params.maxIterations) {
      fprintf(stdout, "\n");
      Warn("max iterations reached at frame %d\n", f_i);
    }

    fprintf(stdout, "Frame %05d %*.2f ms @ %*.2f Hz %03d iters %20s num_points %-*d %s\r",
              f_i-1, 6, tt, 5, (f_i - 1) / total_time,  num_iters,
              ToString(result.keyFramingReason).c_str(), 8, 0/*vo.numPointsAtLevel()*/,
              desc_name.c_str());
    fflush(stdout);


    num_iterations.push_back(num_iters);
    time_ms.push_back(tt);
    trajectory.push_back(result.pose);
  }

  std::string path_file = prefix + "_" + desc_name + "_path.txt";
  std::string time_file = prefix + "_" + desc_name + "_time.txt";
  std::string num_iters_file = prefix + "_" + desc_name + "_iters.txt";

  std::cout << "path_file: " << path_file << std::endl;
  std::cout << "time_file: " << time_file << std::endl;
  std::cout << "iters_file: " << num_iters_file << std::endl;

  trajectory.writeCameraPath(path_file);
  WriteContainer(time_file, time_ms);
  WriteContainer(num_iters_file, num_iterations);

  fprintf(stdout, "\n");
}

//
// Evaluation script to run all the descriptors on the Tsukuba data and write
// results to a file for further analysis in matlab
//
int main(int argc, char** argv)
{
  ProgramOptions options(argv[0]);
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba_eval.cfg", "config_file")
      ("output,o", "", "output prefix")
      ("numframes,n", int(700), "number of frames to process").parse(argc, argv);

  auto conf_fn    = options.get<std::string>("config");
  auto prefix     = options.get<std::string>("output");
  auto numframes  = options.get<int>("numframes");

  std::vector<std::string> desc_names
  {
    "Intensity",
    "IntensityAndGradient",
    "DescriptorFields",
    "Latch",
    "Laplacian",
    "CentralDifference",
    "BitPlanes"
  };

  for(const auto& dname : desc_names) {
    Run(conf_fn, prefix, numframes, dname);
  }

  return 0;
}
