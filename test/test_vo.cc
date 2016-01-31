#include "test/data_loader.h"
#include "bpvo/vo.h"
#include "bpvo/trajectory.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  AlgorithmParameters params("../conf/tsukuba.cfg");
  std::cout << params << std::endl;


  TsukubaDataLoader data_loader;
  auto calib = data_loader.calibration();

  VisualOdometry vo(calib.K, calib.baseline, data_loader.imageSize(), params);

  UniquePointer<ImageFrame> frame;

  int f_i = 1, k = 0;
  while( nullptr != (frame = data_loader.getFrame(f_i++)) && k != 'q')
  {
    cv::imshow("image", frame->image());
    cv::imshow("disparity", colorizeDisparity(frame->disparity()));
    k = cv::waitKey(5) & 0xff;

    auto result = vo.addFrame(frame->image().ptr<uint8_t>(),
                              frame->disparity().ptr<float>());

    printf("num points %d\n", vo.numPointsAtLevel(0));

    if(f_i > 2)
      break;
  }

  return 0;
}
