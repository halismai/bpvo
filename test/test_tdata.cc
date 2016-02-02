#include "bpvo/template_data_.h"
#include "bpvo/warps.h"
#include "bpvo/channels.h"
#include "bpvo/timer.h"

#include "test/data_loader.h"

#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>

#include <limits>
#include <algorithm>

using namespace bpvo;

//typedef RawIntensity ChannelsType;
typedef BitPlanes ChannelsType;

int main()
{
  TsukubaDataLoader data_loader;
  auto calib = data_loader.calibration();

  typedef TemplateData_<ChannelsType, RigidBodyWarp> TData;
  TData data(calib.K, calib.baseline, 0);

  auto frame = data_loader.getFrame(1);

  data.setData( ChannelsType(frame->image()), frame->disparity());

  printf("got %d points\n", data.numPoints());

  std::vector<float> residuals;
  std::vector<uint8_t> valid;
  ChannelsType channels(frame->image());
  Matrix44 pose(Matrix44::Identity());
  data.computeResiduals(channels, pose, residuals, valid);

  auto r_norm = Eigen::VectorXf::Map(residuals.data(), residuals.size()).norm();
  printf("r_norm %f (%g) %g %0.2f%% valid\n", r_norm, r_norm / residuals.size(),
         std::numeric_limits<float>::epsilon(),
         100.0 * (std::count(valid.begin(), valid.end(), true) / (float) valid.size()));

  return 0;

  {
    printf("Timing stuff\n");
    {
      auto t = TimeCode(1000, [&]() { data.setData(ChannelsType(frame->image()), frame->disparity()); });
      printf("setData time %0.2f ms\n", t);
    }

    {
      auto t = TimeCode(1000, [&]() { data.computeResiduals(channels, pose, residuals, valid);  });
      printf("computeResiduals time %0.2f ms\n", t);
    }
  }


  return 0;
}

