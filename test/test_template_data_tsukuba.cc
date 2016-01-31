#include <bpvo/vo.h>
#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

static const char* IMAGE_FILENAME =
"/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png";
static const char* DMAP_FILENAME =
"/home/halismai/data/NewTsukubaStereoDataset/groundtruth/disparity_maps/left/tsukuba_disparity_L_00001.png";

int main()
{
  auto I = cv::imread(IMAGE_FILENAME, cv::IMREAD_GRAYSCALE);
  THROW_ERROR_IF( I.empty(), "could not read image" );

  auto D = cv::imread(DMAP_FILENAME);
  THROW_ERROR_IF( D.empty(), "could not read image");
  D.convertTo(D, CV_32FC1);

  bpvo::Matrix33 K;
  K << 615, 0, 320, 0, 615, 240, 0, 0, 1;
  float b = 0.1;

  bpvo::AlgorithmParameters params;
  params.numPyramidLevels = -1;
  params.maxIterations = 10;
  params.verbosity = bpvo::VerbosityType::kSilent;
  bpvo::VisualOdometry vo(K, b, {480,640}, params);

  std::cout << vo.addFrame(I.ptr<uint8_t>(), D.ptr<float>()) << std::endl;
  std::cout << vo.addFrame(I.ptr<uint8_t>(), D.ptr<float>()) << std::endl;

  auto t = bpvo::TimeCode(100, [&]() { vo.addFrame(I.ptr<uint8_t>(), D.ptr<float>()); });
  printf("addFrame time %0.2f ms [%d points]\n", t, vo.numPointsAtLevel(0));
  printf("timer per iteration %0.2f ms\n", t / params.maxIterations);
  printf("done\n");


  return 0;
}
