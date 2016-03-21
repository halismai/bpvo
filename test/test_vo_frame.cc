#include "bpvo/vo_frame.h"
#include "bpvo/timer.h"
#include "bpvo/template_data.h"
#include "utils/tsukuba_dataset.h"

using namespace bpvo;

int main()
{
  AlgorithmParameters p;
  p.numPyramidLevels = 4;
  p.descriptor = DescriptorType::kIntensity;
  p.parameterTolerance = 1e-6;

  TsukubaSyntheticDataset dataset("../conf/tsukuba.cfg");

  VisualOdometryFrame vo_frame(dataset.calibration().K, dataset.calibration().baseline, p);
  auto frame = dataset.getFrame(0);

  Timer timer;
  int nf = 100;
  for(int i = 0; i < nf; ++i) {
    vo_frame.setData(frame->image(), frame->disparity());
    vo_frame.setTemplate();
    //vo_frame.setDataAndTemplate(frame->image(), frame->disparity());
  }

  auto tt = (double) timer.stop().count();
  printf("done in [%0.2f Hz]\n", nf / (tt / 1000.0));

  return 0;
}
