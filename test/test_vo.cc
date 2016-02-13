#include "utils/data_loader.h"
#include "utils/bounded_buffer.h"
#include "utils/program_options.h"
#include "utils/viz.h"

#include "bpvo/vo.h"
#include "bpvo/trajectory.h"
#include "bpvo/debug.h"
#include "bpvo/timer.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main(int argc, char* argv[])
{
  ProgramOptions options;
  options
      ("config,c", "../conf/tsukuba.cfg", "config file")
      ("output,o", "output.txt", "trajectory output file")
      ("numframes,n", int(100), "number of frames").parse(argc, argv);

  auto max_frames = options.get<int>("numframes");
  auto conf_fn = options.get<std::string>("config");

  AlgorithmParameters params(conf_fn);
  std::cout << params << std::endl;
  std::cout << "---------------------------------------\n";

  UniquePointer<DataLoader> data_loader(new TsukubaDataLoader);
  typename DataLoaderThread::BufferType image_buffer(32);
  auto calib = data_loader->calibration();
  VisualOdometry vo(calib.K, calib.baseline, data_loader->imageSize(), params);


  DataLoaderThread data_loader_thread(std::move(data_loader), image_buffer);

  Trajectory trajectory;
  typename DataLoaderThread::BufferType::value_type frame;

  double total_time = 0.0;
  int k = 0, f_i = 1;
  while('q' != (k = cv::waitKey(5) & 0xff) && f_i < max_frames) {
    if(image_buffer.pop(&frame)) {
      ++f_i;
      Timer timer;
      auto result = vo.addFrame(frame->image().ptr<uint8_t>(),
                              frame->disparity().ptr<float>());
      total_time += timer.stop().count() / 1000.0;

      trajectory.push_back(result.pose);

      cv::imshow("image", frame->image());
      cv::imshow("disparity", colorizeDisparity(frame->disparity()));
    }
  }
  Info("done %0.2f Hz\n", f_i / total_time);

  {
    auto output_fn = options.get<std::string>("output");
    if(!output_fn.empty()) {
      Info("Writing trajectory to %s\n", output_fn.c_str());
      std::ofstream ofs(output_fn);
      if(ofs.is_open()) {
        for(size_t i = 0; i < trajectory.size(); ++i) {
          const auto T = trajectory[i];
          ofs << T(0,3) << " " << T(1,3) << " " << T(2,3) << std::endl;
        }
      }
    }
  }

  return 0;
}

