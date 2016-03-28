#include "bpvo/utils.h"
#include "bpvo/timer.h"
#include "bpvo/config_file.h"

#include "utils/dataset.h"
#include "utils/program_options.h"
#include "utils/sgm.h"
#include "utils/rsgm.h"
#include "utils/viz.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  cv::Mat I0 = cv::imread(
      "/home/halismai/data/kitti/dataset/sequences/00/image_0/000000.png",
      cv::IMREAD_GRAYSCALE);

  cv::Mat I1 = cv::imread(
      "/home/halismai/data/kitti/dataset/sequences/00/image_1/000000.png",
      cv::IMREAD_GRAYSCALE);

  THROW_ERROR_IF( I0.empty() || I1.empty(), "failed to read images" );

  SgmStereo::Config sgm_config;
  sgm_config.numberOfDisparities = 128;

  cv::Mat D0, D1;

  SgmStereo stereo(sgm_config);
  stereo.compute(I0, I1, D0);

  RSGM rsgm;
  rsgm.compute(I0, I1, D1);

  cv::Mat dimg0, dimg1;

  overlayDisparity(I0, D0, dimg0, 0.5, 0.0, 128.0);
  overlayDisparity(I1, D1, dimg1, 0.5, 0.0, 128.0);

  cv::imshow("D0", dimg0);
  cv::imshow("D1", dimg1);

  auto t = TimeCode(10, [&]() { stereo.compute(I0, I1, D0); });
  printf("time SGM  %f\n", t);

  t = TimeCode(10, [&]() { rsgm.compute(I0, I1, D0); });
  printf("time rSGM %f\n", t);

  cv::waitKey(0);

  return 0;
}
