#include "utils/data_loader.h"
#include "utils/bounded_buffer.h"
#include "utils/program_options.h"

#include "bpvo/vo.h"
#include "bpvo/debug.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;


int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("config,c", "../conf/tsukuba.cfg", "config file")
      ("output,o", "output.txt", "trajectory output file")
      ("numframes,n", int(100), "number of frames to process")
      ("dontshow,x", "do not show images")
      .parse(argc, argv);

  auto max_frames = options.get<int>("numframes");
  auto conf_fn = options.get<std::string>("config");
  auto do_show = !options.hasOption("dontshow");

  auto data_loader = DataLoader::FromConfig(conf_fn);
  typename DataLoaderThread::BufferType image_buffer(16);

  AlgorithmParameters params(conf_fn);
  std::cout << "------- AlgorithmParameters -------" << std::endl;
  std::cout << params << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  auto vo = VisualOdometry(data_loader.get(), params);

  Trajectory trajectory;
  SharedPointer<ImageFrame> frame;

  int f_i = data_loader->firstFrameNumber();
  DataLoaderThread data_loader_thread(std::move(data_loader), image_buffer);

  double total_time = 0.0;

  while(f_i < max_frames) {
    if(image_buffer.pop(&frame)) {
      Timer timer;
      auto result = vo.addFrame(frame->image().ptr<uint8_t>(),
                                frame->disparity().ptr<float>());
      double tt = timer.stop().count();
      total_time += (tt / 1000.0);
      f_i += 1;
      trajectory.push_back(result.pose);

      fprintf(stdout, "Frame %05d time %0.2f ms [%0.2f Hz]\r", f_i-1, tt,
              (f_i - 1) / total_time);
      fflush(stdout);

      if(do_show) {
        cv::imshow("image", frame->image());
        cv::imshow("disparity", colorizeDisparity(frame->disparity()));
        int k = 0xff & cv::waitKey(5);
        if(k == 'q' || k == 27)
          break;
      }
    }
  }

  Info("Processed %d frames @ %0.2f Hz\n", f_i, f_i / total_time);

  {
    auto output_fn = options.get<std::string>("output");
    if(!output_fn.empty()) {
      Info("Writing trajectory to %s\n", output_fn.c_str());
      if(!trajectory.writeCameraPath(output_fn)) {
        Warn("failed to write trajectory to %s\n", output_fn.c_str());
      }
    }
  }

  return 0;
}

