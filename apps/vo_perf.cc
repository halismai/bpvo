#include "utils/bounded_buffer.h"
#include "utils/program_options.h"
#include "utils/dataset.h"

#include "bpvo/config.h"
#include "bpvo/trajectory.h"
#include "bpvo/utils.h"
#include "bpvo/vo_kf.h"
#include "bpvo/timer.h"

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

class Vo
{
 public:
  inline Vo(const Matrix33& K, float b, ImageSize imsize, AlgorithmParameters p)
      : _impl(make_unique<VisualOdometryWithKeyFraming>(K, b, imsize, p)) {}

  template <class DatasetPtr> inline
  Vo(const DatasetPtr* p, AlgorithmParameters params)
    : _impl(make_unique<VisualOdometryWithKeyFraming>(
            p->calibration().K,
            p->calibration().baseline,
            p->imageSize(),
            params)) {}

  inline Result addFrame(const uint8_t* I, const float* D)
  {
    return _impl->addFrame(I, D);
  }

  inline Result addFrame(const DatasetFrame* frame)
  {
    return addFrame(frame->image().ptr<const uint8_t>(),
                    frame->disparity().ptr<const float>());
  }

 protected:
  UniquePointer<VisualOdometryWithKeyFraming> _impl;
}; // Vo

int main(int argc, char** argv)
{
  ProgramOptions options("vo_perf");
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba.cfg", "config file")
      ("output,o", "", "prefix to store results for later analysis")
      ("numframes,n", int(1000), "number of frames to process")
      ("dontshow,x", "do not show the image").parse(argc, argv);

  const auto conf_fn = options.get<std::string>("config");
  const auto max_frames = options.get<int>("numframes");
  const auto do_show = !options.hasOption("dontshow");
  auto dataset = Dataset::Create(conf_fn);

  AlgorithmParameters params(conf_fn);
  auto maxTestLevel = params.maxTestLevel;
  auto vo = Vo(dataset.get(), params);


  Trajectory trajectory;
  UniquePointer<DatasetFrame> frame;

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
  }

  return 0;
}

