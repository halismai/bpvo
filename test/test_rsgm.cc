#include "utils/rsgm.h"
#include "utils/viz.h"
#include "bpvo/utils.h"
#include "bpvo/timer.h"

#include <opencv2/highgui/highgui.hpp>

int main()
{
  cv::Mat I0 = cv::imread("/home/halismai/data/kitti/dataset/sequences/00/image_0/000000.png",
                          cv::IMREAD_GRAYSCALE);
  cv::Mat I1 = cv::imread("/home/halismai/data/kitti/dataset/sequences/00/image_1/000000.png",
                          cv::IMREAD_GRAYSCALE);
  cv::Mat D;

  THROW_ERROR_IF( I0.empty() || I1.empty(), "Failed to read images");

  RSGM::Config rsgm_config;
  RSGM rsgm(rsgm_config);

  rsgm.compute(I0, I1, D);

  cv::Mat dimg;
  bpvo::colorizeDisparity(D, dimg, 0, 128);
  //bpvo::overlayDisparity(I0, D, dimg, 0.5, 0.0, 128.0);
  cv::imshow("disparity", dimg);
  cv::waitKey(0);


  printf("Timing\n");
  auto t = bpvo::TimeCode(10, [&]() { rsgm.compute(I0, I1, D); });
  printf("time: %0.2f ms\n", t);

  return 0;
}
