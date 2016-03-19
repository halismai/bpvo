#include "utils/bounded_buffer.h"
#include "utils/program_options.h"
#include "utils/dataset_loader_thread.h"

#include "bpvo/config.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"
#include "bpvo/utils.h"

#include "bpvo/vo_kf.h"

#include <iostream>
#include <fstream>

#if defined(WITH_PROFILER)
#include <gperftools/profiler.h>
#endif

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

class Vo
{
 public:
  inline Vo(const Matrix33& K, float b, ImageSize imsize, AlgorithmParameters p)
      : _impl(make_unique<VisualOdometryWithKeyFraming>(K, b, imsize, p)) {}

  inline Result addFrame(const uint8_t* I, const float* D)
  {
    return _impl->addFrame(I, D);
  }

  inline int numPoints() const
  {
    return _impl->numPointsAtLevel();
  }

 protected:
  UniquePointer<VisualOdometryWithKeyFraming> _impl;
}; // Vo

int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba.cfg", "config file")
      ("output,o", "output.txt", "trajectory output file")
      ("numframes,n", int(100), "number of frames to process")
      ("dontshow,x", "don't show images")
      .parse(argc, argv);

  //cv::setNumThreads(1); // disable opencv threading

  auto conf_fn = options.get<std::string>("config");
  auto max_frames = options.get<int>("numframes");
  auto dataset = Dataset::Create(conf_fn);
  auto do_show = !options.hasOption("dontshow");

  AlgorithmParameters params(conf_fn);
  auto maxTestLevel = params.maxTestLevel;
  auto vo = Vo(dataset->calibration().K, dataset->calibration().baseline,
               dataset->imageSize(), params);

  typename DatasetLoaderThread::BufferType image_buffer(32);
  DatasetLoaderThread data_loader_thread(std::move(dataset), image_buffer);


  Trajectory trajectory;

  double total_time = 0.0;
  int f_i = 0;
  UniquePointer<DatasetFrame> frame;
  while(f_i < max_frames)
  {
    if(image_buffer.pop(&frame, 2))
    {
      if(!frame)
      {
        printf("no more data\n");
        break;
      }

      if(do_show) {
        cv::imshow("image", frame->image());
        int k = 0xff & cv::waitKey(5);
        if('q' == k)
          break;
      }

      const auto I_ptr = frame->image().ptr<const uint8_t>();
      const auto D_ptr = frame->disparity().ptr<const float>();

      Timer timer;
      const auto result = vo.addFrame(I_ptr, D_ptr);
      double tt = timer.stop().count();
      total_time += (tt / 1000.0);

      f_i += 1;
      trajectory.push_back(result.pose);

      int num_iters = result.optimizerStatistics[maxTestLevel].numIterations;
      if(num_iters == params.maxIterations)
      {
        printf("\n");
        Warn("maximum iterations reached %d\n", params.maxIterations);
      }

#if 1
      fprintf(stdout, "Frame %05d %*.2f ms @ %*.2f Hz %03d iters %20s num_points %-*d\r",
              f_i-1, 6, tt, 5, (f_i - 1) / total_time,  num_iters,
              ToString(result.keyFramingReason).c_str(), 8, vo.numPoints());
      fflush(stdout);
#endif
    }
  }

  fprintf(stdout, "\n");
  Info("Processed %d frames @ %0.2f Hz\n", f_i, f_i / total_time);

  {
    auto output_fn = options.get<std::string>("output");
    if(!output_fn.empty()) {
      Info("Writing trajectory to %s\n", output_fn.c_str());
      if(!trajectory.writeCameraPath(output_fn)) {
        Warn("failed to write trajectory to %s\n", output_fn.c_str());
      }
    }

    std::ofstream ofs("poses.txt");
    for(size_t i = 0;i < trajectory.size(); ++i) {
      ofs << trajectory[i] << std::endl;
    }
    ofs.close();
  }

  Info("done\n");
  return 0;
}

