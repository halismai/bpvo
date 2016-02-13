#include "bpvo/vo.h"
#include "utils/viz.h"

#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Geometry>

#include <iostream>

static const char* IMG_FMT = "/media/halismai/bat/tunnel/2015-05-19_bruceton/robotLogData-4/image%06d.pgm";
static const char* DMAP_FMT = "/media/halismai/bat/tunnel/2015-05-19_bruceton/robotLogData-4/image%06d-disparity.pgm";

using namespace bpvo;

struct Frame
{
  cv::Mat I;
  cv::Mat D;

  inline bool get(int f_i)
  {
    char filename_buf[1024];
    snprintf(filename_buf, 1024, IMG_FMT, f_i);
    I = cv::imread(filename_buf, cv::IMREAD_GRAYSCALE);

    snprintf(filename_buf, 1024, DMAP_FMT, f_i);
    D = cv::imread(filename_buf, cv::IMREAD_UNCHANGED);
    if(!D.empty())
      D.convertTo(D, CV_32FC1, 1.0/16.0, 0.0);

    return !I.empty() && !D.empty();
  }

  inline ImageSize imageSize() const {
    return ImageSize(I.rows, I.cols);
  }

  inline const uint8_t* imagePointer() const { return I.ptr<uint8_t>(); }
  inline const float* disparityPointer() const { return D.ptr<float>(); }
}; // Frame

int main()
{
  int frame_start = 1180;
  int num_frames = 10;

  Frame frame;
  if(!frame.get(frame_start)) {
    printf("could not read data\n");
    return 1;
  }

  Matrix33 K;
  K << 604.5975,    0,  507.0000,
         0,  604.5975,  272.0000,
         0,         0,    1.0000;

  AlgorithmParameters params;
  params.numPyramidLevels = 3;
  params.maxIterations = 100;
  params.parameterTolerance = 1e-6;
  params.functionTolerance = 1e-5;
  params.minTranslationMagToKeyFrame = 0.0;
  params.verbosity = VerbosityType::kSilent;
  VisualOdometry vo(K, 0.0702, frame.imageSize(), params);

  auto result = vo.addFrame(frame.imagePointer(), frame.disparityPointer());
  std::vector<Matrix44> poses;
  poses.push_back(result.pose);

  for(int i = 1; i < num_frames; ++i) {
    if(!frame.get(frame_start + i)) {
      printf("could not read data\n");
      return 1;
    }

    cv::imshow("image", overlayDisparity(frame.I, frame.D, 0.75f));
    int k = cv::waitKey(5) & 0xff;
    if(k == ' ') k = cv::waitKey(0) & 0xff;
    if(k == 'q') break;

    result = vo.addFrame(frame.imagePointer(), frame.disparityPointer());
    poses.push_back( poses.back() * result.pose.inverse() );

    printf("Frame %d num_iters %d\n", i, result.optimizerStatistics.front().numIterations);
  }


  for(size_t i = 0; i < poses.size(); ++i) {
    std::cout << poses[i].block<3,1>(0,3).transpose() << std::endl;
  }


  return 0;
}
