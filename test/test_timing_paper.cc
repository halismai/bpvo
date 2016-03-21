#include "bpvo/vo_kf2.h"
#include "bpvo/trajectory.h"
#include "bpvo/timer.h"
#include "bpvo/debug.h"

#include "utils/dataset.h"
#include "utils/program_options.h"

#include <iostream>
#include <fstream>

using namespace bpvo;

template <typename T> static inline
void WriteVector(std::string filename, const std::vector<T>& v)
{
  std::ofstream ofs(filename);
  for(auto v_i : v)
    ofs << v_i << "\n";

  ofs.close();
}

int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("config,c", "", "config file")
      ("numframes,n", int(500), "number of frames")
      ("output,o", "", "output prefix to store results")
      .parse(argc, argv);

  auto dataset = Dataset::Create("/home/halismai/code/bpvo/conf/tsukuba.cfg");

  auto conf_fn = options.get<std::string>("config");
  auto output_prefix = options.get<std::string>("output");
  auto numframes = options.get<int>("numframes");

  std::cout << "config_file:   " << conf_fn << std::endl;
  std::cout << "output_prefix: " << output_prefix << std::endl;

  AlgorithmParameters params(conf_fn);
  VisualOdometryKeyFraming vo(dataset->calibration().K, dataset->calibration().baseline,
                              dataset->imageSize(), params);

  std::vector<int> num_iters;
  std::vector<int> kf_inds;
  std::vector<double> time_ms;

  num_iters.reserve(numframes);
  kf_inds.reserve(numframes);
  time_ms.reserve(numframes);

  Result result;
  Trajectory trajectory;
  double total_time = 0.0;
  for(int i = 0; i < numframes; ++i)
  {
    auto frame = dataset->getFrame(i);

    Timer timer;
    result = vo.addFrame(frame->image().ptr<uint8_t>(), frame->disparity().ptr<float>());
    auto t_ms = timer.stop().count();
    total_time += t_ms / 1000.0; // seconds

    num_iters.push_back( result.optimizerStatistics[params.maxTestLevel].numIterations );
    time_ms.push_back(t_ms);
    if(result.isKeyFrame)
      kf_inds.push_back(i);

    trajectory.push_back( result.pose );

    fprintf(stdout, "Frame %d/%d\r", i, numframes);
    fflush(stdout);
  }

  Info("Done @ %0.2f Hz\n", numframes / total_time);

  trajectory.writeCameraPath(output_prefix + "_path.txt");
  WriteVector(output_prefix + "_iters.txt", num_iters);
  WriteVector(output_prefix + "_time_ms.txt", time_ms);
  WriteVector(output_prefix + "_kf_inds.txt", kf_inds);

  return 0;
}

