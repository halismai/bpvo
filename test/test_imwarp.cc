#include "bpvo/types.h"
#include "bpvo/warps.h"
#include "bpvo/interp_util.h"
#include "bpvo/timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <algorithm>

using namespace bpvo;

int main()
{
  Matrix44 T(Matrix44::Identity());

  cv::Mat I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png", cv::IMREAD_GRAYSCALE);

  cv::Mat D = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/groundtruth/disparity_maps/left/tsukuba_disparity_L_00001.png", cv::IMREAD_UNCHANGED);

  D.convertTo(D, CV_32FC1, 1.0 / 16.0);
  I.convertTo(I, CV_32FC1, 1.0);

  Matrix33 K;
  K  << 615.0, 0.0, 320.0,
     0.0, 615.0, 240.0,
     0.0, 0.0, 1.0;
  float b = 0.1;


  T(0,3) = 0.001;
  T(1,3) = 0.002;
  T(2,3) = 0.003;
  RigidBodyWarp warp(K, b);
  warp.setPose(T);

  typename EigenAlignedContainer<Point>::type points;
  for(int y = 1, i=0; y < D.rows-1; ++y)
    for(int x = 1; x < D.cols-1; ++x, ++i)
      points.emplace_back(warp.makePoint(x, y, D.at<float>(y,x)));

  points.erase(points.end()-1, points.end());

  BilinearInterp<> interp;
  interp.init(warp, points, D.rows, D.cols);

  auto I_ptr = I.ptr<float>();
  std::vector<float> Iw;
  for(size_t i = 0; i < points.size(); ++i) {
    Iw.push_back( interp(I_ptr, i) );
  }

  float f = std::count(interp.valid().begin(), interp.valid().end(), 1) /
      (float) interp.valid().size();
  printf("got %f\n", f);

  interp.initFast(warp, points, D.rows, D.cols);
  std::vector<float> Iw2;
  for(size_t i = 0; i < points.size(); ++i) {
    Iw2.push_back( interp(I_ptr, i) );
  }

  f = std::count(interp.valid().begin(), interp.valid().end(), 1) /
      (float) interp.valid().size();
  printf("got %f\n", f);

  std::vector<float> diff;
  int n_bad = 0;
  for(size_t i = 0; i < Iw.size(); ++i) {
    diff.push_back(std::abs(Iw[i] - Iw2[i]));
    if(diff.back() > 1e-3) {
      printf("%zu %f %f %f\n", i, Iw[i], Iw2[i], diff.back());
      ++n_bad;
    }
  }


  for(size_t i = Iw.size()-3; i < Iw.size(); ++i) {
    printf("%f %f\n", Iw[i], Iw2[i]);
  }

  printf("diff %f\n", Eigen::VectorXf::Map(diff.data(), diff.size()).norm());
  printf("percent off %0.2f%%\n", 100.0f*n_bad/(float) diff.size());


  {
    std::vector<float> ii( points.size() );
    auto code = [&]() {
      interp.init(warp, points, D.rows,  D.cols);
#pragma omp parallel for
      for(size_t i = 0; i < points.size(); ++i)
        ii[i] = interp(I_ptr, i);
    };

    auto t = TimeCode(100, code);
    printf("time %f\n", t);
  }

  {
    std::vector<float> ii( points.size() );
    auto code = [&]() {
      interp.initFast(warp, points, D.rows,  D.cols);
#pragma omp parallel for
      for(size_t i = 0; i < points.size(); ++i)
        ii[i] = interp(I_ptr, i);
    };

    auto t = TimeCode(100, code);
    printf("time %f\n", t);

  }

  return 0;
}




