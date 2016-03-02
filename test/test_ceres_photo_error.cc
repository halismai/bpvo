#include <bpvo/debug.h>

#if !defined(WITH_CERES)
int main() { Fatal("compile with ceres"); }
#else

#include <bpvo/trajectory.h>
#include <dmv/photo_vo.h>
#include <utils/tsukuba_dataset.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/cubic_interpolation.h>

#include <sophus/se3.hpp>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>


using namespace bpvo;

int main()
{
  UniquePointer<Dataset> dataset = UniquePointer<Dataset>(
      new TsukubaSyntheticDataset("../conf/tsukuba.cfg"));

  Eigen::Matrix<double,3,3> K = dataset->calibration().getIntrinsics().cast<double>();

  dmv::PhotoVoBase::Config config;
  config.nonMaxSuppRadius = 1;
  UniquePointer<dmv::PhotoVoBase> vo = UniquePointer<dmv::PhotoVoBase>(
      new dmv::PhotoVo(K, 0.1, config));

  Trajectory trajectory;
  auto f1 = dataset->getFrame(0);
  vo->setTemplate(f1->image(), f1->disparity());

  for(int i = 1; i < 10; ++i)
  {
    Info("Frame %d\n", i);

    f1 = dataset->getFrame(i);
    auto result = vo->estimatePose(f1->image());
    vo->setTemplate(f1->image(), f1->disparity());

    trajectory.push_back(result.T.cast<float>());

    /*
    cv::imshow("image", f1->image());
    int k = 0xff & cv::waitKey(1);
    if(k == 'q') break;
    */
  }

  std::string output_fn("output.txt");
  if(!trajectory.writeCameraPath(output_fn)) {
    Warn("failed to write trajectory to %s\n", output_fn.c_str());
  }

  return 0;
}

#endif
