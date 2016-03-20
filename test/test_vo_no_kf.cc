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
  p.descriptor = DescriptorType::kBitPlanes;
  p.lossFunction = LossFunctionType::kHuber;
  p.parameterTolerance = 1e-6;
  p.verbosity = VerbosityType::kSilent;

  TsukubaSyntheticDataset dataset("../conf/tsukuba.cfg");
  VisualOdometryNoKeyFraming vo(dataset.calibration().K, dataset.calibration().baseline,
                                dataset.imageSize(), p);

  int nf = 500;
  Trajectory trajectory;
  Timer timer;
  for(int i = 0; i < nf; ++i)
  {
    Info("Frame %d\n", i);
    auto frame = dataset.getFrame(i);
    auto result = vo.addFrame(frame->image().ptr<uint8_t>(), frame->disparity().ptr<float>());
    trajectory.push_back(result.pose);
  }

  double tt = timer.stop().count() / 1000.0;
  printf("done @ %0.2f Hz\n", nf / tt);

  trajectory.writeCameraPath("output.txt");

  return 0;
}

