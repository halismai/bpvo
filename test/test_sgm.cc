#include "bpvo/utils.h"
#include "bpvo/timer.h"
#include "bpvo/config_file.h"

#include "utils/dataset.h"
#include "utils/program_options.h"
#include "utils/sgm.h"

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

  SgmStereo stereo(sgm_config);

  cv::Mat D;
  stereo.compute(I0, I1, D);

  FILE* fp = fopen("D.txt", "w");
  for(int y = 0; y < D.rows; ++y)
  {
    for(int x = 0; x < D.cols; ++x)
    {
      fprintf(fp, "%f ", D.at<float>(y,x));
    }

    fprintf(fp, "\n");
  }
  fclose(fp);
  printf("done ... writing to file\n");

  auto t = TimeCode(10, [&]() { stereo.compute(I0, I1, D); });
  printf("time %f\n", t);

  return 0;
}
