#include "bpvo/vo_kf2.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"
#include "bpvo/debug.h"
#include "utils/program_options.h"
#include "utils/tsukuba_dataset.h"

#include <iostream>
#include <fstream>

using namespace bpvo;

template <typename T> static inline
void WriteVector(std::string filename, const std::vector<T>& v)
{
  std::ofstream ofs(filename);
  for(auto v_i : v)
    ofs << v_i << "\n";

  ofs.close();
}

int main()
{
  cv::setNumThreads(1);

  AlgorithmParameters p;
  p.numPyramidLevels = 3;
  p.descriptor = DescriptorType::kBitPlanes;
  p.parameterTolerance = 1e-6;
  p.functionTolerance = 1e-5;

  p.verbosity = VerbosityType::kSilent;
  p.lossFunction = LossFunctionType::kL2;

  p.minTranslationMagToKeyFrame = 100.05;
  p.minRotationMagToKeyFrame = 200.5;

  p.maxFractionOfGoodPointsToKeyFrame = 0.75;

  if(p.lossFunction == LossFunctionType::kL2) {
    p.minTranslationMagToKeyFrame = 0.1;
    p.minRotationMagToKeyFrame = 5.0;
  }

  p.sigmaPriorToCensusTransform = 0.75;
  p.sigmaBitPlanes = 1.6;

  TsukubaSyntheticDataset dataset("../conf/tsukuba.cfg");
  VisualOdometryKeyFraming vo(dataset.calibration().K, dataset.calibration().baseline,
                              dataset.imageSize(), p);

  std::vector<int> num_iters;
  std::vector<double> time_ms;
  std::vector<int> kf_inds;
  double total_time = 0.0;
  int nf = 300;
  Trajectory trajectory;
  for(int i = 0; i < nf; ++i)
  {
    auto frame = dataset.getFrame(i);

    Timer timer;
    auto result = vo.addFrame(frame->image().ptr<uint8_t>(), frame->disparity().ptr<float>());
    double tt = timer.stop().count();
    time_ms.push_back(tt);
    num_iters.push_back(result.optimizerStatistics[p.maxTestLevel].numIterations);
    total_time += tt / 1000.0;
    trajectory.push_back(result.pose);

    Info("Frame %d %d iters keyframe:%s\n", i,
         result.optimizerStatistics.front().numIterations, result.isKeyFrame ? "YES" : "NO");

    if(result.isKeyFrame)
      kf_inds.push_back(i);
  }

  printf("done @ %0.2f Hz\n", nf / total_time);
  printf("%0.2f%% KF\n", 100.0 * kf_inds.size() / (float) nf);

  //trajectory.writeCameraPath("output.txt");
  {
    std::ofstream ofs("output.txt");
    for(size_t i = 0; i < kf_inds.size(); ++i)
    {
      ofs << trajectory[ kf_inds[i] ].block<3,1>(0,3).transpose() << std::endl;
    }

    // the last one in
    if(kf_inds.back() != nf - 1)
      ofs << trajectory[nf - 1].block<3,1>(0,3).transpose() << std::endl;
  }

  WriteVector("kf_inds.txt", kf_inds);
  WriteVector("num_iters.txt", num_iters);
  WriteVector("time_ms.txt", time_ms);

  return 0;
}

