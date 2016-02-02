#include "bpvo/template_data_.h"
#include "bpvo/warps.h"
#include "bpvo/channels.h"
#include "bpvo/timer.h"

#include "test/data_loader.h"

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  TsukubaDataLoader data_loader;
  auto calib = data_loader.calibration();

  typedef TemplateData_<BitPlanes, RigidBodyWarp> TData;
  TData data(calib.K, calib.baseline, 0);

  auto frame = data_loader.getFrame(1);

  data.setData( BitPlanes(frame->image()), frame->disparity());

  printf("got %d points\n", data.numPoints());

  std::vector<float> residuals;
  std::vector<uint8_t> valid;
  BitPlanes channels(frame->image());
  Matrix44 pose(Matrix44::Identity());
  data.computeResiduals(channels, pose, residuals, valid);


  {
    printf("Timing stuff\n");
    {
      auto t = TimeCode(100, [&]() { data.setData(BitPlanes(frame->image()), frame->disparity()); });
      printf("setData time %0.2f ms\n", t);
    }

    {
      auto t = TimeCode(100, [&]() { data.computeResiduals(channels, pose, residuals, valid);  });
      printf("computeResiduals time %0.2f ms\n", t);
    }
  }


  return 0;
}

