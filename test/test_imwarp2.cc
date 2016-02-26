#include "bpvo/types.h"
#include "bpvo/rigid_body_warp.h"
#include "bpvo/interp_util.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

typedef EigenAlignedContainer<Point>::type PointVector;

template <class Warp> static inline typename BilinearInterp<float>::CoeffsVector
TestInterpBasic(const Warp& warp, const PointVector& points, const cv::Mat& image)
{
  BilinearInterp<float> interp;
  interp.init(warp, points, image.rows, image.cols);

  for(int i = 72; i < 72 + 9; ++i)
    printf("%d ", interp.indices()[i]);
  printf("\n");

  return interp.getInterpCoeffs();
}

template <class Warp> static inline typename BilinearInterp<float>::CoeffsVector
TestInterpFast(const Warp& warp, const PointVector& points, const cv::Mat& image)
{
  BilinearInterp<float> interp;
  interp.initFast(warp, points, image.rows, image.cols);

  for(int i = 72; i < 72 + 9; ++i)
    printf("%d ", interp.indices()[i]);
  printf("\n");

  return interp.getInterpCoeffs();
}


int main()
{
  cv::Mat I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png", cv::IMREAD_GRAYSCALE);

  cv::Mat D = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/groundtruth/disparity_maps/left/tsukuba_disparity_L_00001.png", cv::IMREAD_UNCHANGED);

  D.convertTo(D, CV_32FC1, 1.0 / 16.0);
  I.convertTo(I, CV_32FC1, 1.0);

  Matrix33 K;
  K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;
  RigidBodyWarp warp(K, 0.1);

  Matrix44 T(Matrix44::Identity());
  warp.setPose(T);
  T(0,3) = 0.01;

  PointVector points;
  for(int y = 1; y < D.rows - 2; ++y)
    for(int x = 1; x < D.cols - 2; ++x) {
      points.push_back(warp.makePoint(x, y, D.at<float>(y,x)));
    }

  std::cout<< "P: " << points[72].transpose() << std::endl;
  printf("%f %f %f %f\n", points[72][0], points[72][1], points[72][2], points[72][3]);

  auto c1 = TestInterpBasic(warp, points, I);
  auto c2 = TestInterpFast(warp, points, D);

  int num_bad = 0;
  for(size_t i = 0; i < c1.size(); ++i)
  {
    auto d = (c1[i] - c2[i]).norm();
    if(d > 1e-3) {
      std::cout << "error at " << i << std::endl;
      std::cout << c1[i].transpose() << std::endl;
      std::cout << c2[i].transpose() << std::endl;
      //std::cout << points[i].transpose() << std::endl;
      ++num_bad;
      if(num_bad > 4)
        exit(0);
    }
  }

  printf("num bad %d %f\n", num_bad, num_bad / (double) points.size());

  return 0;
}
