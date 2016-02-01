#include "test/data_loader.h"
#include "bpvo/vo.h"
#include "bpvo/trajectory.h"
#include "bpvo/debug.h"
#include "bpvo/timer.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  AlgorithmParameters params("../conf/tsukuba.cfg");
  std::cout << params << std::endl;

  TsukubaDataLoader data_loader;
  auto calib = data_loader.calibration();

  VisualOdometry vo(calib.K, calib.baseline, data_loader.imageSize(), params);

  Trajectory trajectory;
  UniquePointer<ImageFrame> frame;

  double total_time = 0.0;
  int f_i = 1, k = 0;
  while( nullptr != (frame = data_loader.getFrame(f_i++)) && k != 'q' && f_i < 125)
  {
    dprintf("FRAME %d\n", f_i);

    cv::imshow("image", frame->image());
    cv::imshow("disparity", colorizeDisparity(frame->disparity()));
    k = cv::waitKey(5) & 0xff;

    Timer timer;
    auto result = vo.addFrame(frame->image().ptr<uint8_t>(),
                              frame->disparity().ptr<float>());
    total_time += timer.stop().count() / 1000.0;

    trajectory.push_back(result.pose);
  }

  Info("done %0.2f Hz\n", f_i / total_time);

  std::ofstream ofs("output.txt");
  if(ofs.is_open()) {
    for(size_t i = 0; i < trajectory.size(); ++i) {
      const auto T = trajectory[i];
      ofs << T(0,3) << " " << T(1,3) << " " << T(2,3) << std::endl;
    }

    ofs.close();
  }

  return 0;
}
