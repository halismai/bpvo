#include "bpvo/vo_no_kf.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"
#include "bpvo/debug.h"
#include "utils/tsukuba_dataset.h"

#include <iostream>

using namespace bpvo;

int main()
{
  AlgorithmParameters p;
  p.numPyramidLevels = 3;
  p.descriptor = DescriptorType::kIntensity;
  p.lossFunction = LossFunctionType::kHuber;
  p.parameterTolerance = 1e-6;
  p.functionTolerance = 1e-5;

  p.verbosity = VerbosityType::kSilent;
  p.lossFunction = LossFunctionType::kHuber;

  TsukubaSyntheticDataset dataset("../conf/tsukuba.cfg");
  VisualOdometryNoKeyFraming vo(dataset.calibration().K, dataset.calibration().baseline,
                                dataset.imageSize(), p);

  double total_time = 0.0;
  int nf = 300;
  Trajectory trajectory;
  for(int i = 0; i < nf; ++i)
  {
    auto frame = dataset.getFrame(i);

    Timer timer;
    auto result = vo.addFrame(frame->image().ptr<uint8_t>(), frame->disparity().ptr<float>());
    total_time += timer.stop().count() / 1000.0;
    trajectory.push_back(result.pose);

    Info("Frame %d %d iters\n", i, result.optimizerStatistics.front().numIterations);
  }

  printf("done @ %0.2f Hz\n", nf / total_time);

  trajectory.writeCameraPath("output.txt");

  return 0;
}

